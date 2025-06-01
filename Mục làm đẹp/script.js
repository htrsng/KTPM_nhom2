const foods = [
  { name: "Cơm trắng", calories: 130 },
  { name: "Ức gà luộc", calories: 165 },
  { name: "Trứng luộc", calories: 78 },
  { name: "Sữa chua", calories: 100 }
];

let tdee = 0;
let consumedCalories = 0;
let foodLog = JSON.parse(localStorage.getItem('foodLog')) || {};
let calorieChart;

// Lấy ngày hiện tại dạng YYYY-MM-DD
const getCurrentDate = () => {
  const today = new Date();
  return today.toISOString().split('T')[0];
};

function init() {
  // Khởi tạo foodLog cho ngày hiện tại nếu chưa có
  if (!foodLog[getCurrentDate()]) {
    foodLog[getCurrentDate()] = [];
  }
  loadFoodOptions();
  setupEventListeners();
  initChart();      // <-- Đặt trước
  loadFoodLog();    // <-- Đặt sau
}

function setupEventListeners() {
  document.getElementById('calculateTDEEButton').addEventListener('click', calculateTDEE);
  document.getElementById('addFoodButton').addEventListener('click', addFood);
  document.getElementById('addManualCaloriesButton').addEventListener('click', () => {
    const manualCalories = parseInt(document.getElementById('manualCalories').value);
    if (isNaN(manualCalories) || manualCalories <= 0) {
      Toastify({ text: "Vui lòng nhập calo hợp lệ", backgroundColor: "red", duration: 3000 }).showToast();
      return;
    }
    const entry = { name: "Thêm thủ công", calories: manualCalories };
    addEntry(entry);
  });
  document.getElementById('resetFoodLogButton').addEventListener('click', () => {
    if (confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử ngày hôm nay?")) {
      foodLog[getCurrentDate()] = [];
      saveFoodLog();
      loadFoodLog();
    }
  });
  document.getElementById('themeToggle').addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
  });
}

// Load danh sách món ăn vào select
function loadFoodOptions() {
  const foodSelect = document.getElementById('foodSelect');
  foods.forEach(food => {
    const option = document.createElement('option');
    option.value = food.name;
    option.textContent = `${food.name} (${food.calories} calo)`;
    foodSelect.appendChild(option);
  });
}

// Tính TDEE theo công thức Mifflin-St Jeor
function calculateTDEE() {
  const gender = document.getElementById('gender').value;
  const age = parseInt(document.getElementById('age').value);
  const height = parseInt(document.getElementById('height').value);
  const weight = parseFloat(document.getElementById('weight').value);
  const activity = parseFloat(document.getElementById('activity').value);

  if (isNaN(age) || isNaN(height) || isNaN(weight)) {
    Toastify({ text: "Vui lòng điền đầy đủ thông tin", backgroundColor: "red", duration: 3000 }).showToast();
    return;
  }

  let bmr;
  if (gender === 'male') {
    bmr = 10 * weight + 6.25 * height - 5 * age + 5;
  } else {
    bmr = 10 * weight + 6.25 * height - 5 * age - 161;
  }

  tdee = Math.round(bmr * activity);
  document.getElementById('tdeeResult').textContent = `TDEE của bạn là: ${tdee} calo/ngày`;

  updateProgress();
  updateSuggestion();
}

// Thêm món ăn từ danh sách
function addFood() {
  const selectedName = document.getElementById('foodSelect').value;
  const selectedFood = foods.find(food => food.name === selectedName);
  if (!selectedFood) {
    Toastify({ text: "Vui lòng chọn món ăn", backgroundColor: "orange", duration: 3000 }).showToast();
    return;
  }
  addEntry(selectedFood);
}

// Thêm mục vào log và cập nhật
function addEntry(entry) {
  foodLog[getCurrentDate()].push(entry);
  saveFoodLog();
  loadFoodLog();
  Toastify({
    text: "Thêm thành công!",
    style: { background: "linear-gradient(to right, #00b09b, #96c93d)" }
  }).showToast();
}

// Hiển thị danh sách đã ăn
function loadFoodLog() {
  const list = document.getElementById('foodList');
  list.innerHTML = '';
  const todayLog = foodLog[getCurrentDate()] || [];
  consumedCalories = 0;
  todayLog.forEach(item => {
    const li = document.createElement('li');
    li.textContent = `${item.name} - ${item.calories} calo`;
    list.appendChild(li);
    consumedCalories += item.calories;
  });
  updateProgress();
  updateChart();
}

// Lưu log vào localStorage
function saveFoodLog() {
  localStorage.setItem('foodLog', JSON.stringify(foodLog));
}

// Cập nhật thanh tiến trình và trạng thái
function updateProgress() {
  const progress = tdee > 0 ? Math.min((consumedCalories / tdee) * 100, 100) : 0;
  const progressBar = document.getElementById('progressBar');
  progressBar.style.width = `${progress}%`;
  progressBar.textContent = `${Math.round(progress)}%`;

  // Đổi màu nếu vượt quá TDEE
  if (tdee > 0 && consumedCalories > tdee) {
    progressBar.classList.remove('bg-blue-600');
    progressBar.classList.add('bg-red-600');
    
  } else {
    progressBar.classList.remove('bg-red-600');
    progressBar.classList.add('bg-blue-600');
    
  }

  const status = document.getElementById('calorieStatus');
  status.textContent = `Đã tiêu thụ ${consumedCalories} / ${tdee || '...'} calo`;

  // Hiển thị số món ăn
  const todayLog = foodLog[getCurrentDate()] || [];
  document.getElementById('foodCount').textContent = `Số món: ${todayLog.length}`;
}

// Gợi ý giảm cân
function updateSuggestion() {
  const suggestion = document.getElementById('suggestion');
  if (tdee > 0) {
    suggestion.textContent = `Để giảm cân hiệu quả, bạn có thể ăn ít hơn khoảng ${Math.round(tdee * 0.2)} calo so với TDEE (${tdee}) mỗi ngày.`;
  }
}

// Vẽ biểu đồ
function initChart() {
  const ctx = document.getElementById('calorieChart').getContext('2d');
  calorieChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        label: 'Lượng calo mỗi ngày',
        data: [],
        backgroundColor: 'rgba(59, 130, 246, 0.7)',
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
  updateChart();
}

// Cập nhật dữ liệu biểu đồ
function updateChart() {
  const dates = Object.keys(foodLog).sort();
  calorieChart.data.labels = dates;
  calorieChart.data.datasets[0].data = dates.map(date => {
    return foodLog[date].reduce((sum, item) => sum + item.calories, 0);
  });
  calorieChart.update();
}

// Khởi tạo sau khi load
document.addEventListener('DOMContentLoaded', init);