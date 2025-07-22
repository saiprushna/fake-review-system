const singleReviewForm = document.getElementById('singleReviewForm');
const singleReviewInput = document.getElementById('singleReviewInput');
const singleReviewResult = document.getElementById('singleReviewResult');

if (singleReviewForm) {
    singleReviewForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const review = singleReviewInput.value.trim();
        if (!review) return;
        singleReviewResult.textContent = 'Analyzing...';

        try {
            const response = await fetch('https://fake-review-system.vercel.app/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review })
            });
            const data = await response.json();
            if (data.prediction) {
                singleReviewResult.innerHTML = `<b>Result:</b> <span style="color:${data.prediction === 'genuine' ? '#16a34a' : '#dc2626'}">${data.prediction.toUpperCase()}</span>`;
            } else {
                singleReviewResult.textContent = 'Error: ' + (data.error || 'Unknown error');
            }
        } catch (err) {
            singleReviewResult.textContent = 'Error: Could not connect to server.';
        }
    });
} 
