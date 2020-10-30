const tableHeaders = document.getElementById('tableHeaders');
const tableBody = document.getElementById('tableBody');
const arrhythmiaType = document.getElementById('arrhythmiaType');
const arrInterval = document.getElementById('arrInterval');
const ecgChart = document.getElementById('ecgChart');
let chartGlobal;

let arrhythmiaTypes;

(function populateTableBody() {
    let html = ``;
    let classes = [];
    for (arr of arrhythmias) {
        classes.push(arr.class_);
    }
    let classes_unique = [...new Set(classes)];
    arrhythmiaTypes = [...classes_unique];
    let count = [];
    let temp;
    for (let i = 0; i < classes_unique.length; i++) {
        temp = 0
        for (let j = 0; j < classes.length; j++) {
            if (classes_unique[i] === classes[j]) {
                temp++;
            }
        }
        count.push(temp);
    }

    for (let i = 0; i < classes_unique.length; i++) {
        html += `
        <tr>
            <th scope="col">${classes_unique[i]}</th>
            <td>${count[i]}<td>
        </tr>
        `;
    }

    tableBody.innerHTML = html;
})();

(function populateArrhythmiaType() {
    let html = `<option disabled selected>Choose arrhythmia type</option>`;
    for (let arr of arrhythmiaTypes) {
        html += `<option>${arr}</option>`;
    }
    arrhythmiaType.innerHTML = html;
})();

function updateArrIntervals() {
    let html = `<option disabled selected>Choose interval</option>`;
    let currentArr = arrhythmiaType.value;
    let intervalsWithCurrentArr = arrhythmias.filter(e => e.class_ === currentArr);

    for (let inter of intervalsWithCurrentArr) {
        html += `<option value="${inter.qrsindex}">From ${inter.start} to ${inter.end}</option>`;
    }
    arrInterval.innerHTML = html;
}

arrhythmiaType.addEventListener('change', updateArrIntervals);

function updateEcgChart() {
    // Get the interval
    let qrsindex = parseInt(arrInterval.value);
    let interval = arrhythmias.filter(e => e.qrsindex === qrsindex)[0];

    let xLabels = [];
    for (let i = 0; i < interval.signal.length; i++) {
        xLabels.push(interval.start + i);
    }

    chartGlobal.config.data = {
        labels: xLabels,
        datasets: [
            {
                label: `Arrhythmia of type ${interval.class_} from ${interval.start} to ${interval.end}`,
                backgroundColor: "rgb(255, 99, 132)",
                borderColor: "rgb(255, 99, 132)",
                data: interval.signal,
                fill: false,
            },
        ],
    };
    chartGlobal.update();

//    let ctx = ecgChart;
//    let chart = new Chart(ctx, {
//        type: 'line',
//        data: {
//            labels: xLabels,
//            datasets: [
//                {
//                    label: `Arrhythmia of type ${interval.class_} from ${interval.start} to ${interval.end}`,
//                    backgroundColor: "rgb(255, 99, 132)",
//                    borderColor: "rgb(255, 99, 132)",
//                    data: interval.signal,
//                    fill: false,
//                },
//            ],
//        },
//        options: {
//            responsive: false,
//            title: {
//                display: true,
//                text: 'ECG Plot'
//            },
//            tooltips: {
//                mode: 'index',
//                intersect: false,
//            },
//            hover: {
//                mode: 'nearest',
//                intersect: true
//            },
//            scales: {
//                x: {
//                    display: true,
//                    scaleLabel: {
//                        display: true,
//                        labelString: 'Sample'
//                    }
//                },
//                y: {
//                    display: true,
//                    scaleLabel: {
//                        display: true,
//                        labelString: 'Value'
//                    }
//                }
//            }
//        }
//    });
}

arrInterval.addEventListener('change', updateEcgChart);

(function startEcgChart() {
    let ctx = ecgChart;
    chartGlobal = new Chart(ctx, {
        type: 'line',
        data: {},
        options: {
            responsive: false,
            title: {
                display: false,
                text: 'ECG Plot'
            },
            tooltips: {
                mode: 'index',
                intersect: false,
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                x: {
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Sample'
                    }
                },
                y: {
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Value'
                    }
                },
                yAxes: [{
                    ticks: {
                        suggestedMin: -1.8,
                        suggestedMax: 1.8
                    }
                }],
            },
            elements: {
                point:{
                    radius: 0
                }
            },
        }
    });
})();