const analyzeBtn = document.getElementById('analyzeBtn');
const checkArrhythmiasBtn = document.getElementById('checkArrhythmiasBtn');
const signalAnalyzerForm = document.getElementById('signalAnalyzerForm');

function setCheckOfAllArrhythmias(status) {
    $("input[id^='arr']").each(function (i, el) {
         el.checked = status;
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

signalAnalyzerForm.addEventListener('submit', () => {
    analyzeBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...`;
    analyzeBtn.setAttribute("disabled", true);
});

