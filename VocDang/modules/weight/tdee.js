document.addEventListener('DOMContentLoaded', function() {
  // Load saved data from localStorage
  const savedData = JSON.parse(localStorage.getItem('tdeeData'));
  if (savedData) {
    document.getElementById('goal').value = savedData.goal || '';
    document.getElementById('age').value = savedData.age || '';
    document.getElementById('gender').value = savedData.gender || '';
    document.getElementById('height').value = savedData.height || '';
    document.getElementById('weight').value = savedData.weight || '';
    document.getElementById('activity').value = savedData.activity || '';
  }

  // Dark mode toggle
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  if (localStorage.getItem('darkMode') === 'enabled') {
    enableDarkMode();
    darkModeToggle.innerHTML = '<i class="fas fa-sun"></i> Chế độ sáng';
  }

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    
    if (document.body.classList.contains('dark-mode')) {
      localStorage.setItem('darkMode', 'enabled');
      darkModeToggle.innerHTML = '<i class="fas fa-sun"></i> Chế độ sáng';
      enableDarkMode();
    } else {
      localStorage.setItem('darkMode', 'disabled');
      darkModeToggle.innerHTML = '<i class="fas fa-moon"></i> Chế độ tối';
      disableDarkMode();
    }
    
    // Redraw chart with updated colors
    const result = document.getElementById('result');
    if (result.style.display === 'block') {
      const bmr = parseInt(document.getElementById('bmr-value').textContent.split(':')[1].trim().split(' ')[0]);
      const tdee = parseInt(document.getElementById('tdee-value').textContent.split(':')[1].trim().split(' ')[0]);
      const calorieSuggestion = parseInt(document.getElementById('calorie-suggestion').textContent.split(':')[1].trim().split(' ')[0]);
      
      const ctx = document.getElementById('tdee-chart').getContext('2d');
      drawTDEEChart(ctx, bmr, tdee, calorieSuggestion);
    }
  });

  // Save result button
  document.getElementById('save-result')?.addEventListener('click', function() {
    const resultText = `Kết quả TDEE của bạn:
    - BMR: ${document.getElementById('bmr-value').textContent}
    - TDEE: ${document.getElementById('tdee-value').textContent}
    - Khuyến nghị calo: ${document.getElementById('calorie-suggestion').textContent}
    - Gợi ý: ${document.getElementById('suggestions').textContent}`;
    
    localStorage.setItem('tdeeResult', resultText);
    
    // Show notification
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = '<i class="fas fa-check-circle"></i> Đã lưu kết quả!';
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
      notification.classList.remove('show');
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  });
});

function enableDarkMode() {
  document.body.classList.add('dark-mode');
  document.querySelector('.tdee-calculator').classList.add('dark-mode');
  document.querySelector('#result').classList.add('dark-mode');
  document.querySelector('.result-details').classList.add('dark-mode');
  document.querySelector('.navbar').classList.add('dark-mode');
  document.querySelector('footer').classList.add('dark-mode');
}

function disableDarkMode() {
  document.body.classList.remove('dark-mode');
  document.querySelector('.tdee-calculator').classList.remove('dark-mode');
  document.querySelector('#result').classList.remove('dark-mode');
  document.querySelector('.result-details').classList.remove('dark-mode');
  document.querySelector('.navbar').classList.remove('dark-mode');
  document.querySelector('footer').classList.remove('dark-mode');
}

document.getElementById('tdee-form').addEventListener('submit', function(e) {
  e.preventDefault();

  // Get form inputs
  const goal = document.getElementById('goal').value;
  const age = parseInt(document.getElementById('age').value);
  const gender = document.getElementById('gender').value;
  const height = parseInt(document.getElementById('height').value);
  const weight = parseInt(document.getElementById('weight').value);
  const activity = parseFloat(document.getElementById('activity').value);

  // Input validation
  const errorDiv = document.getElementById('error');
  errorDiv.style.display = 'none';
  
  if (!goal || !gender || !activity) {
    showError('Vui lòng chọn tất cả các trường bắt buộc.');
    return;
  }
  if (isNaN(age)) {
    showError('Vui lòng nhập tuổi hợp lệ.');
    return;
  }
  if (age < 1 || age > 120) {
    showError('Tuổi phải từ 1 đến 120.');
    return;
  }
  if (isNaN(height)) {
    showError('Vui lòng nhập chiều cao hợp lệ.');
    return;
  }
  if (height < 50 || height > 250) {
    showError('Chiều cao phải từ 50 đến 250 cm.');
    return;
  }
  if (isNaN(weight)) {
    showError('Vui lòng nhập cân nặng hợp lệ.');
    return;
  }
  if (weight < 20 || weight > 300) {
    showError('Cân nặng phải từ 20 đến 300 kg.');
    return;
  }

  // Save inputs to localStorage
  const userData = { goal, age, gender, height, weight, activity };
  localStorage.setItem('tdeeData', JSON.stringify(userData));

  // Calculate results
  const { bmr, tdee, calorieSuggestion, goalText } = calculateResults(goal, age, gender, height, weight, activity);

  // Display results
  displayResults(bmr, tdee, calorieSuggestion, goalText, goal);

  // Draw chart
  const ctx = document.getElementById('tdee-chart').getContext('2d');
  drawTDEEChart(ctx, bmr, tdee, calorieSuggestion);
});

function showError(message) {
  const errorDiv = document.getElementById('error');
  errorDiv.innerText = message;
  errorDiv.style.display = 'block';
  errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function calculateResults(goal, age, gender, height, weight, activity) {
  // Calculate BMR (Mifflin-St Jeor Equation)
  let bmr;
  if (gender === 'male') {
    bmr = 10 * weight + 6.25 * height - 5 * age + 5;
  } else {
    bmr = 10 * weight + 6.25 * height - 5 * age - 161;
  }
  bmr = Math.round(bmr);

  // Calculate TDEE
  const tdee = Math.round(bmr * activity);

  // Adjust calories based on goal
  let calorieSuggestion;
  let goalText;
  if (goal === 'lose') {
    calorieSuggestion = tdee - 500;
    goalText = 'giảm cân (thiếu hụt 500 calo/ngày)';
  } else if (goal === 'gain') {
    calorieSuggestion = tdee + 500;
    goalText = 'tăng cân (thặng dư 500 calo/ngày)';
  } else {
    calorieSuggestion = tdee;
    goalText = 'duy trì cân nặng';
  }
  calorieSuggestion = Math.round(calorieSuggestion);

  return { bmr, tdee, calorieSuggestion, goalText };
}

function displayResults(bmr, tdee, calorieSuggestion, goalText, goal) {
  document.getElementById('bmr-value').innerHTML = `<span class="result-label">Tỷ lệ trao đổi chất cơ bản (BMR):</span> <span class="result-value">${bmr} calo/ngày</span>`;
  document.getElementById('tdee-value').innerHTML = `<span class="result-label">Tổng năng lượng tiêu hao hàng ngày (TDEE):</span> <span class="result-value">${tdee} calo/ngày</span>`;
  document.getElementById('calorie-suggestion').innerHTML = `<span class="result-label">Lượng calo khuyến nghị để ${goalText}:</span> <span class="result-value highlight-value">${calorieSuggestion} calo/ngày</span>`;
  document.getElementById('explanation').innerText = `BMR là lượng calo cơ thể bạn cần để duy trì các chức năng cơ bản khi nghỉ ngơi. TDEE là tổng lượng calo bạn tiêu thụ mỗi ngày dựa trên mức vận động. Lượng calo khuyến nghị được điều chỉnh dựa trên mục tiêu ${goalText}.`;

  // Display diet/exercise suggestions
  const suggestionsDiv = document.getElementById('suggestions');
  let suggestions = '';
  if (goal === 'lose') {
    suggestions = `
      <div class="suggestion-card">
        <h4><i class="fas fa-utensils"></i> Chế độ ăn</h4>
        <p>Ưu tiên thực phẩm giàu protein (thịt gà, cá, trứng), rau xanh, và giảm tinh bột tinh chế.</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: Salad ức gà, khoai lang luộc, bông cải xanh hấp.</p>
      </div>
      <div class="suggestion-card">
        <h4><i class="fas fa-dumbbell"></i> Luyện tập</h4>
        <p>Kết hợp cardio (chạy bộ, đạp xe 3-4 lần/tuần) và tập tạ nhẹ để duy trì cơ bắp.</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: 30 phút chạy bộ + 20 phút tập tạ nhẹ.</p>
      </div>
    `;
  } else if (goal === 'gain') {
    suggestions = `
      <div class="suggestion-card">
        <h4><i class="fas fa-utensils"></i> Chế độ ăn</h4>
        <p>Tăng lượng protein (thịt bò, whey protein) và carb phức (gạo lứt, yến mạch).</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: Cơm gạo lứt với thịt bò xào, trứng luộc.</p>
      </div>
      <div class="suggestion-card">
        <h4><i class="fas fa-dumbbell"></i> Luyện tập</h4>
        <p>Tập tạ nặng (4-5 lần/tuần) với các bài compound như squat, deadlift, bench press.</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: 5 hiệp squat 8-12 lần, 4 hiệp deadlift 6-8 lần.</p>
      </div>
    `;
  } else {
    suggestions = `
      <div class="suggestion-card">
        <h4><i class="fas fa-utensils"></i> Chế độ ăn</h4>
        <p>Cân bằng protein, carb, và chất béo lành mạnh. Ăn đa dạng thực phẩm.</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: Cá hồi nướng, quinoa, bơ, các loại hạt.</p>
      </div>
      <div class="suggestion-card">
        <h4><i class="fas fa-dumbbell"></i> Luyện tập</h4>
        <p>Duy trì tập luyện 3-4 lần/tuần với cardio nhẹ và bài tập toàn thân.</p>
        <p class="example"><i class="fas fa-lightbulb"></i> Ví dụ: 20 phút cardio + 30 phút tập tạ toàn thân.</p>
      </div>
    `;
  }
  suggestionsDiv.innerHTML = suggestions;

  // Show result section with animation
  const result = document.getElementById('result');
  result.style.display = 'block';
  result.scrollIntoView({ behavior: 'smooth' });
}

function drawTDEEChart(ctx, bmr, tdee, calorieSuggestion) {
  const maxValue = Math.max(bmr, tdee, calorieSuggestion) * 1.2;
  ctx.canvas.width = ctx.canvas.offsetWidth;
  ctx.canvas.height = ctx.canvas.offsetHeight;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Colors based on theme
  const bgColor = document.body.classList.contains('dark-mode') ? '#34495e' : '#f8f9fa';
  const textColor = document.body.classList.contains('dark-mode') ? '#ecf0f1' : '#2c3e50';
  const gridColor = document.body.classList.contains('dark-mode') ? '#2c3e50' : '#ddd';

  // Draw background
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Chart dimensions
  const chartHeight = ctx.canvas.height - 80;
  const chartWidth = ctx.canvas.width - 100;
  const barWidth = 60;
  const gap = 30;
  const startX = 70;
  const baseY = chartHeight + 20;

  // Draw grid lines
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 0.5;
  const gridLines = 5;
  for (let i = 0; i <= gridLines; i++) {
    const y = baseY - (chartHeight * i / gridLines);
    ctx.beginPath();
    ctx.moveTo(startX - 10, y);
    ctx.lineTo(startX + chartWidth, y);
    ctx.stroke();
    
    // Grid values
    ctx.fillStyle = textColor;
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(maxValue * i / gridLines), startX - 15, y + 4);
  }

  // Draw bars
  const barColors = ['#FF6F61', '#007BFF', '#2ecc71'];
  const data = [bmr, tdee, calorieSuggestion];
  const labels = ['BMR', 'TDEE', 'Calo Khuyến Nghị'];
  
  data.forEach((value, index) => {
    const barHeight = (value / maxValue) * chartHeight;
    const x = startX + index * (barWidth + gap);
    
    // Draw bar
    ctx.fillStyle = barColors[index];
    ctx.fillRect(x, baseY - barHeight, barWidth, barHeight);
    
    // Add rounded corners
    ctx.beginPath();
    ctx.moveTo(x, baseY - barHeight + 10);
    ctx.quadraticCurveTo(x, baseY - barHeight, x + 10, baseY - barHeight);
    ctx.lineTo(x + barWidth - 10, baseY - barHeight);
    ctx.quadraticCurveTo(x + barWidth, baseY - barHeight, x + barWidth, baseY - barHeight + 10);
    ctx.lineTo(x + barWidth, baseY);
    ctx.lineTo(x, baseY);
    ctx.closePath();
    ctx.fill();
    
    // Draw value on top
    ctx.fillStyle = textColor;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(value, x + barWidth / 2, baseY - barHeight - 8);
    
    // Draw label
    ctx.fillText(labels[index], x + barWidth / 2, baseY + 20);
  });

  // Draw axes
  ctx.strokeStyle = textColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(startX - 10, baseY);
  ctx.lineTo(startX - 10, 30);
  ctx.lineTo(startX + chartWidth, 30);
  ctx.stroke();
}

// Notification style
const style = document.createElement('style');
style.textContent = `
  .notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    background-color: #2ecc71;
    color: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 1000;
    opacity: 0;
    transform: translateX(100%);
    transition: all 0.3s ease;
  }
  
  .notification.show {
    opacity: 1;
    transform: translateX(0);
  }
  
  .notification i {
    font-size: 1.2rem;
  }
`;
document.head.appendChild(style);