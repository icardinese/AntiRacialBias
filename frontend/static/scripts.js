// scripts.js
document.getElementById('submit-btn').addEventListener('click', function() {
    this.textContent = 'Processing...';
    this.disabled = true;
});
