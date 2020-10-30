const RECOMMENDED_DS = [
    // CHAZAL TEST DATASET
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234',
    // CHAZAL TRAIN DATASET
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230',
];

const checkArrhythmiasBtn = document.getElementById('checkArrhythmiasBtn');
const checkRecordsBtn = document.getElementById('checkRecordsBtn');
const extractSignalsBtn = document.getElementById('extractSignalsBtn');
const extractionParamsForm = document.getElementById('extractionParamsForm');

function setCheckOfAllArrhythmias(status) {
    $("input[id^='arr']").each(function (i, el) {
         el.checked = status;
    });
}

function setCheckOfRecommendedRecords(status) {
    $("input[id^='record']").each(function (i, el) {
        console.log('Hola');
        for (let record of RECOMMENDED_DS) {
            console.log('Record:', record);
            if (el.id.includes(record)) {
                el.checked = status;
                break;
            }
        }
    });
}

checkArrhythmiasBtn.addEventListener('click', () => {
    if (checkArrhythmiasBtn.innerText === 'Check all') {
        checkArrhythmiasBtn.innerText = 'Uncheck all';
        setCheckOfAllArrhythmias(true);
    } else {
        checkArrhythmiasBtn.innerText = 'Check all';
        setCheckOfAllArrhythmias(false);
    }
});

checkRecordsBtn.addEventListener('click', () => {
    if (checkRecordsBtn.innerText === 'Check recommended') {
        checkRecordsBtn.innerText = 'Uncheck recommended';
        setCheckOfRecommendedRecords(true);
    } else {
        checkRecordsBtn.innerText = 'Check recommended';
        setCheckOfRecommendedRecords(false);
    }
});

//extractSignalsBtn.addEventListener('click', () => {
//
//});

extractionParamsForm.addEventListener('submit', () => {
    extractSignalsBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...`;
    extractSignalsBtn.setAttribute("disabled", true);
});
