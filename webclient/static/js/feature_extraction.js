const checkArrhythmiasBtn = document.getElementById('checkArrhythmiasBtn');
const extractFeaturesBtn = document.getElementById('extractFeaturesBtn');
const extractionParamsForm = document.getElementById('extractionParamsForm');

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

extractionParamsForm.addEventListener('submit', () => {
    extractFeaturesBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...`;
    extractFeaturesBtn.setAttribute("disabled", true);
});
