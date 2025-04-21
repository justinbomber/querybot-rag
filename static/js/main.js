// 主要功能已在index.html內實現
// 這裡放置額外的功能

// 頁面載入完成後執行
document.addEventListener('DOMContentLoaded', function() {
    // 檢查頁面連接狀態
    checkServerStatus();
    
    // 添加版本信息
    addVersionInfo();
});

// 檢查伺服器狀態
function checkServerStatus() {
    // 簡單的服務器檢查，使用settings API調用
    fetch('/api/settings')
        .then(response => {
            if (response.ok) {
                console.log('伺服器連接正常');
            } else {
                console.error('伺服器連接異常');
                showConnectionError();
            }
        })
        .catch(error => {
            console.error('伺服器連接失敗', error);
            showConnectionError();
        });
}

// 顯示連接錯誤信息
function showConnectionError() {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger';
    errorDiv.innerHTML = '無法連接到伺服器，請確認伺服器已啟動';
    
    // 將錯誤信息放在頁面頂部
    document.body.prepend(errorDiv);
}

// 添加版本信息
function addVersionInfo() {
    const versionDiv = document.createElement('div');
    versionDiv.className = 'text-center text-muted mt-3';
    versionDiv.innerHTML = 'RAG問答系統 v1.0.0';
    
    // 將版本信息添加到頁面底部
    document.querySelector('.container').appendChild(versionDiv);
} 