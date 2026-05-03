// API Configuration
const API_URL = 'https://heart-disease-api-bbd4.onrender.com';
const API_ENDPOINT = `${API_URL}/predict`;

// DOM Elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsContent = document.getElementById('resultsContent');
const resultsCard = document.getElementById('resultsCard');

// Result Elements
const predictionStatus = document.getElementById('predictionStatus');
const statusText = document.getElementById('statusText');
const riskLevelBadge = document.querySelector('.risk-badge');
const probabilityBar = document.getElementById('probabilityBar');
const probNoDisease = document.getElementById('probNoDisease');
const probDisease = document.getElementById('probDisease');
const confidenceValue = document.getElementById('confidenceValue');
const messageText = document.getElementById('messageText');

// Form validation function
function validateForm(data) {
    const errors = [];
    
    if (data.age < 1 || data.age > 120) {
        errors.push('Age must be between 1 and 120 years');
    }
    if (![0, 1].includes(parseInt(data.sex))) {
        errors.push('Sex must be 0 (Female) or 1 (Male)');
    }
    if (![1, 2, 3, 4].includes(parseInt(data.chestPain))) {
        errors.push('Chest pain type must be 1, 2, 3, or 4');
    }
    if (data.bp < 50 || data.bp > 250) {
        errors.push('Blood pressure must be between 50 and 250 mm Hg');
    }
    if (data.cholesterol < 100 || data.cholesterol > 600) {
        errors.push('Cholesterol must be between 100 and 600 mg/dl');
    }
    if (![0, 1].includes(parseInt(data.fbs))) {
        errors.push('FBS over 120 must be 0 or 1');
    }
    if (![0, 1, 2].includes(parseInt(data.ekg))) {
        errors.push('EKG results must be 0, 1, or 2');
    }
    if (data.maxHr < 60 || data.maxHr > 220) {
        errors.push('Max heart rate must be between 60 and 220 bpm');
    }
    if (![0, 1].includes(parseInt(data.exerciseAngina))) {
        errors.push('Exercise angina must be 0 or 1');
    }
    if (data.stDepression < 0 || data.stDepression > 10) {
        errors.push('ST depression must be between 0 and 10');
    }
    if (![1, 2, 3].includes(parseInt(data.slope))) {
        errors.push('Slope of ST must be 1, 2, or 3');
    }
    if (![0, 1, 2, 3].includes(parseInt(data.vessels))) {
        errors.push('Number of vessels must be 0, 1, 2, or 3');
    }
    if (![3, 6, 7].includes(parseInt(data.thallium))) {
        errors.push('Thallium must be 3, 6, or 7');
    }
    
    return errors;
}

// Show error toast
function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
        <i class="fas fa-times close-toast"></i>
    `;
    
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #e74c3c;
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 1000;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    toast.querySelector('.close-toast').addEventListener('click', () => {
        toast.remove();
    });
    
    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, 5000);
}

// Update UI based on prediction
function updateUI(result) {
    if (result.prediction === 1) {
        predictionStatus.className = 'prediction-status positive';
        predictionStatus.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>⚠️ High Risk - Heart Disease Detected</span>
        `;
    } else {
        predictionStatus.className = 'prediction-status negative';
        predictionStatus.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>✅ Low Risk - No Heart Disease Detected</span>
        `;
    }
    
    riskLevelBadge.className = `risk-badge ${result.risk_level.toLowerCase().replace(' ', '-')}`;
    riskLevelBadge.textContent = result.risk_level;
    
    const diseaseProbability = result.probability * 100;
    probabilityBar.style.width = `${diseaseProbability}%`;
    
    probNoDisease.textContent = `${(result.probability_no_disease * 100).toFixed(1)}%`;
    probDisease.textContent = `${diseaseProbability.toFixed(1)}%`;
    confidenceValue.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    messageText.textContent = result.message;
}

// Show loading animation
function showLoading() {
    loadingContainer.style.display = 'flex';
    resultsContent.style.display = 'none';
    predictBtn.disabled = true;
    predictBtn.style.opacity = '0.6';
    predictBtn.innerHTML = `
        <i class="fas fa-spinner fa-spin"></i>
        <span>Processing...</span>
    `;
}

// Hide loading animation
function hideLoading() {
    loadingContainer.style.display = 'none';
    resultsContent.style.display = 'block';
    predictBtn.disabled = false;
    predictBtn.style.opacity = '1';
    predictBtn.innerHTML = `
        <i class="fas fa-brain"></i>
        <span>Predict Heart Disease Risk</span>
        <i class="fas fa-arrow-right"></i>
    `;
}

// Scroll to results
function scrollToResults() {
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Make API call
async function predictHeartDisease(formData) {
    const requestData = {
        Age: parseInt(formData.age),
        Sex: parseInt(formData.sex),
        Chest_pain_type: parseInt(formData.chestPain),
        BP: parseInt(formData.bp),
        Cholesterol: parseInt(formData.cholesterol),
        FBS_over_120: parseInt(formData.fbs),
        EKG_results: parseInt(formData.ekg),
        Max_HR: parseInt(formData.maxHr),
        Exercise_angina: parseInt(formData.exerciseAngina),
        ST_depression: parseFloat(formData.stDepression),
        Slope_of_ST: parseInt(formData.slope),
        Number_of_vessels_fluro: parseInt(formData.vessels),
        Thallium: parseInt(formData.thallium)
    };
    
    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }
    
    return await response.json();
}

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        age: document.getElementById('age').value,
        sex: document.getElementById('sex').value,
        chestPain: document.getElementById('chestPain').value,
        bp: document.getElementById('bp').value,
        cholesterol: document.getElementById('cholesterol').value,
        fbs: document.getElementById('fbs').value,
        ekg: document.getElementById('ekg').value,
        maxHr: document.getElementById('maxHr').value,
        exerciseAngina: document.getElementById('exerciseAngina').value,
        stDepression: document.getElementById('stDepression').value,
        slope: document.getElementById('slope').value,
        vessels: document.getElementById('vessels').value,
        thallium: document.getElementById('thallium').value
    };
    
    const errors = validateForm(formData);
    if (errors.length > 0) {
        showError(errors[0]);
        return;
    }
    
    showLoading();
    
    try {
        const result = await predictHeartDisease(formData);
        updateUI(result);
        hideLoading();
        scrollToResults();
        
        resultsCard.style.transform = 'scale(1.02)';
        setTimeout(() => {
            resultsCard.style.transform = '';
        }, 300);
        
    } catch (error) {
        console.error('Prediction error:', error);
        hideLoading();
        showError(error.message || 'Failed to get prediction. Please check if the backend server is running.');
    }
});

// Add input validation on blur
const inputs = document.querySelectorAll('input, select');
inputs.forEach(input => {
    input.addEventListener('blur', () => {
        if (input.value === '' || input.value === null) {
            input.style.borderColor = '#e74c3c';
        } else {
            input.style.borderColor = '#ecf0f1';
        }
    });
    
    input.addEventListener('focus', () => {
        input.style.borderColor = '#e74c3c';
    });
});

// Check API health on page load
async function checkAPIHealth() {
    try {
        console.log(`Checking API health at: ${API_URL}/health`);
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API is healthy:', data);
        } else {
            console.warn('API health check failed:', response.status);
            showError('Backend API is not responding. Please try again later.');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        // Don't show error on page load, just log it
        console.warn('Cannot connect to backend server. Make sure API is deployed.');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    
    const formCard = document.querySelector('.form-card');
    if (formCard) {
        formCard.style.opacity = '0';
        formCard.style.transform = 'translateY(20px)';
        setTimeout(() => {
            formCard.style.transition = 'all 0.5s ease';
            formCard.style.opacity = '1';
            formCard.style.transform = 'translateY(0)';
        }, 100);
    }
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .error-toast {
        animation: slideIn 0.3s ease;
    }
    
    .fa-spin {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);