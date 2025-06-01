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
    document.body.classList.add('dark-mode');
    document.querySelector('.tdee-calculator').classList.add('dark-mode');
    document.querySelector('#result').classList.add('dark-mode');
    document.querySelector('.result-details').classList.add('dark-mode');
    document.querySelector('.navbar').classList.add('dark-mode');
    document.querySelector('footer').classList.add('dark-mode');
    darkModeToggle.textContent = '☀️ Chế độ sáng';
  }

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.querySelector('.tdee-calculator').classList.toggle('dark-mode');
    document.querySelector('#result').classList.toggle('dark-mode');
    document.querySelector('.result-details').classList.toggle('dark-mode');
    document.querySelector('.navbar').classList.toggle('dark-mode');
    document.querySelector('footer').classList.toggle('dark-mode');
    if (document.body.classList.contains('dark-mode')) {
      localStorage.setItem('darkMode', 'enabled');
      darkModeToggle.textContent = '☀️ Chế độ sáng';
    } else {
      localStorage.setItem('darkMode', 'disabled');
      darkModeToggle.textContent = '🌙 Chế độ tối';
    }
  });
});

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
  if (!goal || !gender || !activity) {
    errorDiv.innerText = 'Vui lòng chọn tất cả các trường bắt buộc.';
    errorDiv.style.display = 'block';
    return;
  }
  if (age < 1 || age > 120) {
    errorDiv.innerText = 'Tuổi phải từ 1 đến 120.';
    errorDiv.style.display = 'block';
    return;
  }
  if (height < 50 || height > 250) {
    errorDiv.innerText = 'Chiều cao phải từ 50 đến 250 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  if (weight < 20 || weight > 300) {
    errorDiv.innerText = 'Cân nặng phải từ 20 đến 300 kg.';
    errorDiv.style.display = 'block';
    return;
  }
  errorDiv.style.display = 'none';

  // Save inputs to localStorage
  const userData = { goal, age, gender, height, weight, activity };
  localStorage.setItem('tdeeData', JSON.stringify(userData));

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

  // Display results
  document.getElementById('bmr-value').innerText = `Tỷ lệ trao đổi chất cơ bản (BMR): ${bmr} calo/ngày`;
  document.getElementById('tdee-value').innerText = `Tổng năng lượng tiêu hao hàng ngày (TDEE): ${tdee} calo/ngày`;
  document.getElementById('calorie-suggestion').innerText = `Lượng calo khuyến nghị để ${goalText}: ${calorieSuggestion} calo/ngày`;
  document.getElementById('explanation').innerText = `BMR là lượng calo cơ thể bạn cần để duy trì các chức năng cơ bản khi nghỉ ngơi. TDEE là tổng lượng calo bạn tiêu thụ mỗi ngày dựa trên mức vận động. Lượng calo khuyến nghị được điều chỉnh dựa trên mục tiêu ${goalText}.`;

  // Display diet/exercise suggestions
  const suggestionsDiv = document.getElementById('suggestions');
  let suggestions = '';
  if (goal === 'lose') {
    suggestions = `
      <p><strong>Chế độ ăn:</strong> Ưu tiên thực phẩm giàu protein (thịt gà, cá, trứng), rau xanh, và giảm tinh bột tinh chế. Ví dụ: Salad ức gà, khoai lang luộc.</p>
      <p><strong>Luyện tập:</strong> Kết hợp cardio (chạy bộ, đạp xe 3-4 lần/tuần) và tập tạ nhẹ để duy trì cơ bắp.</p>
    `;
  } else if (goal === 'gain') {
    suggestions = `
      <p><strong>Chế độ ăn:</strong> Tăng lượng protein (thịt bò, whey protein) và carb phức (gạo lứt, yến mạch). Ví dụ: Cơm gạo lứt với thịt bò xào.</p>
      <p><strong>Luyện tập:</strong> Tập tạ nặng (4-5 lần/tuần) với các bài squat, deadlift, bench press.</p>
    `;
  } else {
    suggestions = `
      <p><strong>Chế độ ăn:</strong> Cân bằng protein, carb, và chất béo lành mạnh. Ví dụ: Cá hồi nướng, quinoa, bơ.</p>
      <p><strong>Luyện tập:</strong> Duy trì tập luyện 3-4 lần/tuần với cardio nhẹ và bài tập toàn thân.</p>
    `;
  }
  suggestionsDiv.innerHTML = suggestions;

  document.getElementById('result').style.display = 'block';

  // Draw chart
  const ctx = document.getElementById('tdee-chart').getContext('2d');
  drawTDEEChart(ctx, bmr, tdee, calorieSuggestion);
});

function drawTDEEChart(ctx, bmr, tdee, calorieSuggestion) {
  const maxValue = Math.max(bmr, tdee, calorieSuggestion) + 500;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw background
  ctx.fillStyle = document.body.classList.contains('dark-mode') ? '#333' : '#f5f5f5';
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw bars
  const barWidth = 80;
  const gap = 20;
  ctx.fillStyle = '#FF6F61';
  ctx.fillRect(50, 250 - (bmr / maxValue) * 200, barWidth, (bmr / maxValue) * 200);
  ctx.fillStyle = '#007BFF';
  ctx.fillRect(50 + barWidth + gap, 250 - (tdee / maxValue) * 200, barWidth, (tdee / maxValue) * 200);
  ctx.fillStyle = '#28A745';
  ctx.fillRect(50 + 2 * (barWidth + gap), 250 - (calorieSuggestion / maxValue) * 200, barWidth, (calorieSuggestion / maxValue) * 200);

  // Draw labels
  ctx.fillStyle = document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#333';
  ctx.font = '14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('BMR', 50 + barWidth / 2, 270);
  ctx.fillText('TDEE', 50 + barWidth + gap + barWidth / 2, 270);
  ctx.fillText('Calo Khuyến Nghị', 50 + 2 * (barWidth + gap) + barWidth / 2, 270);
  ctx.fillText(`${bmr} calo`, 50 + barWidth / 2, 240 - (bmr / maxValue) * 200);
  ctx.fillText(`${tdee} calo`, 50 + barWidth + gap + barWidth / 2, 240 - (tdee / maxValue) * 200);
  ctx.fillText(`${calorieSuggestion} calo`, 50 + 2 * (barWidth + gap) + barWidth / 2, 240 - (calorieSuggestion / maxValue) * 200);

  // Draw axes
  ctx.beginPath();
  ctx.strokeStyle = document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#333';
  ctx.moveTo(40, 50);
  ctx.lineTo(40, 250);
  ctx.lineTo(350, 250);
  ctx.stroke();

  // Draw legend
  ctx.fillStyle = '#FF6F61';
  ctx.fillRect(300, 20, 15, 15);
  ctx.fillStyle = '#007BFF';
  ctx.fillRect(300, 40, 15, 15);
  ctx.fillStyle = '#28A745';
  ctx.fillRect(300, 60, 15, 15);
  ctx.fillStyle = document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#333';
  ctx.font = '12px Arial';
  ctx.textAlign = 'left';
  ctx.fillText('BMR', 320, 30);
  ctx.fillText('TDEE', 320, 50);
  ctx.fillText('Calo Khuyến Nghị', 320, 70);
}