const getClassifierForm = document.getElementById('getClassifierForm');
const getClassifierBtn = document.getElementById('getClassifierBtn');
//const checkArrhythmiasBtn = document.getElementById('checkArrhythmiasBtn');

//function setCheckOfAllArrhythmias(status) {
//    $("input[id^='arr']").each(function (i, el) {
//         el.checked = status;
//    });
//}
//
//checkArrhythmiasBtn.addEventListener('click', () => {
//    if (checkArrhythmiasBtn.innerText === 'Check all') {
//        checkArrhythmiasBtn.innerText = 'Uncheck all';
//        setCheckOfAllArrhythmias(true);
//    } else {
//        checkArrhythmiasBtn.innerText = 'Check all';
//        setCheckOfAllArrhythmias(false);
//    }
//});

getClassifierForm.addEventListener('submit', () => {
    getClassifierBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
        Loading...`;
    getClassifierBtn.setAttribute("disabled", true);
});