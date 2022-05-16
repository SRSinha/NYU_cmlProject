import os
import string 
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
import os
import argparse

def run():
    bestmodel  = ''
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')                    
    args = parser.parse_args()
    print("Running for Batch Size ", args.batch_size)
    print("Running for Epoch ", args.epochs)
   
    def load_file(filepath, device, MAX_VOCAB_SIZE = 200):
        tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()
        TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=100)
        LABEL = data.Field(sequential=False, use_vocab=False)
    
        tv_datafields = [("text", TEXT), ("label", LABEL)]
        
        train, valid, test = data.TabularDataset.splits(path=filepath,
                                                        train="train1000.csv", validation="val200.csv",
                                                        test="test100.csv", format="csv",
                                                        skip_header=True, fields=tv_datafields)
        TEXT.build_vocab(train, max_size = MAX_VOCAB_SIZE)
        
        train_iter = data.BucketIterator(train, device=device, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                        sort_within_batch=False, repeat=False)
        valid_iter = data.BucketIterator(valid, device=device, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                        sort_within_batch=False, repeat=False)
        test_iter = data.BucketIterator(test, device=device, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                        sort_within_batch=False, repeat=False)
        print("Construct iterator success...")
        return TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device ", device)
    TEXT, LABEL, train, valid, test, train_iter, valid_iter, test_iter = load_file('imdb1000', device)
    print(TEXT.vocab.freqs.most_common(20))
    print(TEXT.vocab.itos[:10])

    class SentimentModel(nn.Module):
        def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, text):
            embedded = self.embedding(text)
            output, hidden = self.rnn(embedded)
            assert torch.equal(output[-1,:,:], hidden.squeeze(0))
            
            return self.fc(hidden.squeeze(0))

    INPUT_DIM = len(TEXT.vocab)
    print(INPUT_DIM)
    EMBEDDING_DIM = 100
    EMBEDDING_DIM = 20
    HIDDEN_DIM = 256
    HIDDEN_DIM = 40
    OUTPUT_DIM = 1

    model = SentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def binary_accuracy(preds, y):
        '''
        Return accuracy per batch ..
        '''
        
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        
        return acc

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()
        
        for i, batch in enumerate(iterator):
            optimizer.zero_grad()

            predictions = model(batch.text).squeeze(1)
            
            # note we must transform the batch.label into float or we will get an error later.
            loss = criterion(predictions, batch.label.float())
            acc = binary_accuracy(predictions, batch.label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if i % 200 == 199:
                print(f"[{i}/{len(iterator)}] : epoch_acc: {epoch_acc / len(iterator):.2f}")
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                # prediction [batch_size]
                predictions = model(batch.text).squeeze(1)
                
                loss = criterion(predictions, batch.label.float())
                
                acc = binary_accuracy(predictions, batch.label)
            
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
        return epoch_loss / len(iterator),  epoch_acc / len(iterator)

    import time

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time  / 60)
        elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
        return  elapsed_mins, elapsed_secs

    N_epoches = 1
    best_valid_loss = float('inf')

    for epoch in range(N_epoches):
        
        start_time = time.time()
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bestmodel= 'Sentiment-model'+'_'+str(args.epochs)+'_'+str(args.batch_size)+'.pt'
            torch.save(model.state_dict(), 'Sentiment-model'+'_'+str(args.epochs)+'_'+str(args.batch_size)+'.pt')
            
        print(f'Epoch:  {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain  Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')
        # print(f'\tValid  Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load(bestmodel))
    # test_loss, test_acc = evaluate(model, test_iter, criterion)
    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


if __name__ =="__main__":
    torch.cuda.empty_cache()
    run()