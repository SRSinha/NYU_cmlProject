function deleteNote(noteId) {
  fetch("/delete-note", {
    method: "POST",
    body: JSON.stringify({ noteId: noteId }),
  }).then((_res) => {
    window.location.href = "/";
  });
}

$('#image').on('change', function () {
  var fullPath = $(this).val();
  var fileName = fullPath.replace(/^.*[\\\/]/, '')
  $(this).next('.custom-file-label').html(fileName);
})

$(document).ready(function () {
  $('a.active').removeClass('active');
  $('a[href="' + location.pathname + '"]').closest('a').addClass('active');
});

function loading(){
  $("#loading").show();
  $("#content").hide();       
}


const labels = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
];

const data = {
  labels: labels,
  datasets: [{
    label: 'My First dataset',
    backgroundColor: 'rgb(255, 99, 132)',
    borderColor: 'rgb(255, 99, 132)',
    data: [0, 10, 5, 2, 20, 30, 45],
  }]
};

const config = {
  type: 'line',
  data: data,
  options: {}
};

function createChart()
{
  const myChart = new Chart(
    document.getElementById('myChart'),
    config
  );
}
