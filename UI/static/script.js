document.addEventListener('DOMContentLoaded', () => {
    console.log('Flask static files are working!');
    const p = document.querySelector('p');
    p.textContent += ' (updated by script.js)';
});

