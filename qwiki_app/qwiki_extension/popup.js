document.addEventListener('DOMContentLoaded', function (tab) {
    chrome.tabs.getSelected(null, function (tab) {
        // console.log('Turning ' + tab.url + ' red!');
        $.get("http://127.0.0.1:8000/api?url=" + tab.url, function (res) {
            document.getElementById("result").innerHTML = res;
        });
    });
});