from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, send_file, send_from_directory
from flask_login import login_required, current_user
from shelljob import proc
import subprocess
import time
import os
from PIL import Image
from .models import Note, Sentiment
from . import db
import json
import random
import math

import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app_root = os.path.dirname(os.path.abspath(__file__))
root_batch = '32'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def inference(img):
    model = Net()
    target = os.path.join(app_root, 'mnt')
    destination = '/'.join([target, 'mnist_cnn.pt'])
    model.load_state_dict(torch.load(destination, map_location='cpu'))
    model.eval()

    img = img.to(device)
    try:
        output = model(img)
        index = output.data.numpy().argmax()
    except:
        print("Something went wrong")
    else:
        return str(index)


views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        note = request.form.get('note')

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added!', category='success')
    return render_template("home.html", user=current_user)


@views.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    if request.method == 'POST':
        time.sleep(15)
        batch = request.form.get('batch')
        epochs = request.form.get('epochs')
        target = os.path.join(app_root, 'static/scripts')
        destination = '/'.join([target, 'script.sh'])
        destination += ' '
        print(destination)
        command = ['sh ' + destination + batch]
        cmd = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(cmd.communicate())
        target = os.path.dirname(app_root)
        target = os.path.join(target, batch)
        file_name = 'nvprof-gpuop-' + batch + '.log'
        destination = '/'.join([target, file_name])
        # target = os.path.join(app_root, 'logs')
        # file_name = 'nvprof'
        # destination = '/'.join([target, file_name])
        line_number = search_string_in_file(
            destination, "dram_read_transactions")
        dram_read_transactions = get_data_from_line_number(
            destination, line_number)
        line_number = search_string_in_file(
            destination, "dram_write_transactions")
        dram_write_transactions = get_data_from_line_number(
            destination, line_number)
        line_number = search_string_in_file(destination, "FLOPS")
        FLOPS = get_data_from_line_number(destination, line_number)
        global root_batch
        root_batch = batch
        return render_template("train.html", user=current_user, batch=batch, epochs=epochs, read=dram_read_transactions, write=dram_write_transactions, flops=FLOPS, showResult=True, filename=file_name)
    return render_template("train.html", user=current_user)


@views.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        return render_template("profile.html", user=current_user)
    return render_template("profile.html", user=current_user)


@views.route('/sentiment', methods=['GET', 'POST'])
@login_required
def sentiment():
    if request.method == 'POST':
        sentence = request.form.get('sentence')
        score = sentiment_scores(sentence)
        new_sentiment = Sentiment(
            sentence=sentence, score=score, user_id=current_user.id)
        db.session.add(new_sentiment)
        db.session.commit()
        return render_template("sentiment.html", user=current_user, sentence=sentence, score=score, showScore=True)
    return render_template("sentiment.html", user=current_user)


@views.route('/mnist', methods=['GET', 'POST'])
@login_required
def mnist():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return render_template("mnist.html", user=current_user)
        file_name = file.filename or ''
        target = os.path.join(app_root, 'static/downloads')
        destination = '/'.join([target, file_name])
        file.save(destination)
        image = Image.open(file)
        image = transform(image)
        image = image.unsqueeze(0)
        ans = inference(image)
        return render_template("mnist.html", user=current_user, filename=file_name, answer=ans)
    return render_template("mnist.html", user=current_user)


@views.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='downloads/' + filename), code=301)


@views.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    # print(filename)
    # target = os.path.join(app_root, 'static/uploads')
    # filenames = 'nvprof-gpuop-64.log'
    target = os.path.dirname(app_root)
    global root_batch
    target = os.path.join(target, root_batch)
    print(root_batch)
    print(target)
    return send_from_directory(target, filename, as_attachment=True)


@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"

    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"

    else:
        return "Neutral"


def search_string_in_file(file_name, string_to_search):
    line_number = 0
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                return line_number
    return line_number


def get_data_from_line_number(file_name, line_number):
    file = open(file_name)
    content = file.readlines()
    return content[line_number]
