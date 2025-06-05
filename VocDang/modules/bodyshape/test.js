document.addEventListener('DOMContentLoaded', function () {
  // Load saved data from localStorage
  const savedData = JSON.parse(localStorage.getItem('shapeData'));
  if (savedData) {
    document.getElementById('shoulder').value = savedData.shoulder || '';
    document.getElementById('chest').value = savedData.chest || '';
    document.getElementById('waist').value = savedData.waist || '';
    document.getElementById('hip').value = savedData.hip || '';
    document.getElementById('height').value = savedData.height || '';
    document.getElementById('gender').value = savedData.gender || 'female';
  }

  // Load saved result if exists
  const savedResult = localStorage.getItem('savedShapeResult');
  if (savedResult) {
    displayResult(JSON.parse(savedResult));
  }
  // Dark mode toggle
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  if (localStorage.getItem('darkMode') === 'enabled') {
    enableDarkMode();
    darkModeToggle.textContent = '☀️ Chế độ sáng';
  }

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.querySelector('.shape-analyzer').classList.toggle('dark-mode');
    
    if (document.body.classList.contains('dark-mode')) {
      localStorage.setItem('darkMode', 'enabled');
      darkModeToggle.textContent = '☀️ Chế độ sáng';
    } else {
      localStorage.setItem('darkMode', 'disabled');
      darkModeToggle.textContent = '🌙 Chế độ tối';
    }
  });


  // Shape icon click event
  document.querySelectorAll('.shape-icon').forEach(icon => {
    icon.addEventListener('click', function () {
      const shape = this.getAttribute('data-shape');
      const gender = document.getElementById('gender').value;
      showShapeInfo(shape, gender);
    });
  });

  // Save result button
  document.getElementById('save-result')?.addEventListener('click', function () {
    const currentResult = localStorage.getItem('currentShapeResult');
    if (currentResult) {
      localStorage.setItem('savedShapeResult', currentResult);
      this.innerHTML = '<i class="fas fa-check"></i> Đã lưu';
      setTimeout(() => {
        this.innerHTML = '<i class="fas fa-bookmark"></i> Lưu kết quả';
      }, 2000);
    }
  });

  function enableDarkMode() {
  document.body.classList.add('dark-mode');
  document.querySelector('.shape-analyzer').classList.add('dark-mode');
  document.querySelector('.navbar').classList.add('dark-mode');
}

  // Load styles.json
  fetch('styles.json')
    .then(response => response.json())
    .then(data => {
      window.bodyShapes = data; // Lưu vào biến toàn cục để sử dụng
    })
    .catch(error => {
      console.error('Error loading styles.json:', error);
    });
});

document.getElementById('shape-form').addEventListener('submit', function (e) {
  e.preventDefault();

  // Get form inputs
  const shoulder = parseFloat(document.getElementById('shoulder').value);
  const chest = parseFloat(document.getElementById('chest').value);
  const waist = parseFloat(document.getElementById('waist').value);
  const hip = parseFloat(document.getElementById('hip').value);
  const height = parseFloat(document.getElementById('height').value);
  const gender = document.getElementById('gender').value;

  // Input validation
  const errorDiv = document.getElementById('error');
  errorDiv.style.display = 'none';

  const inputs = [
    { value: shoulder, name: 'vai', min: 20, max: 100 },
    { value: chest, name: 'ngực', min: 50, max: 150 },
    { value: waist, name: 'eo', min: 40, max: 120 },
    { value: hip, name: 'hông', min: 50, max: 150 },
    { value: height, name: 'chiều cao', min: 100, max: 250 },
  ];

  for (const input of inputs) {
    if (isNaN(input.value) || input.value < input.min || input.value > input.max) {
      errorDiv.innerText = `Số đo ${input.name} phải từ ${input.min} đến ${input.max} cm.`;
      errorDiv.style.display = 'block';
      return;
    }
  }

  // Save inputs to localStorage
  const shapeData = { shoulder, chest, waist, hip, height, gender };
  localStorage.setItem('shapeData', JSON.stringify(shapeData));

  // Calculate body shape based on ratios
  const shoulderToHipRatio = shoulder / hip;
  const waistToHipRatio = waist / hip;
  const waistToShoulderRatio = waist / shoulder;

  let shape;
  if (shoulderToHipRatio >= 1.05 && waistToHipRatio > 0.85) {
    shape = 'apple';
  } else if (shoulderToHipRatio < 0.95 && waistToHipRatio > 0.8) {
    shape = 'pear';
  } else if (shoulderToHipRatio >= 0.95 && shoulderToHipRatio <= 1.05 && waistToHipRatio <= 0.75) {
    shape = 'hourglass';
  } else if (shoulderToHipRatio > 1.05 && waistToHipRatio <= 0.75) {
    shape = 'inverted-triangle';
  } else {
    shape = 'rectangle';
  }

  // Display results
  displayResult({ ...window.bodyShapes[shape], gender });

  // Save current result
  localStorage.setItem('currentShapeResult', JSON.stringify({ ...window.bodyShapes[shape], gender }));
});

function displayResult(shapeData) {
  const { title, description, gender } = shapeData;
  const recommendations = shapeData[`recommendations_${gender}`] || [];
  const avoid = shapeData[`avoid_${gender}`] || [];

  document.getElementById('shape-title').innerText = title;
  document.getElementById('shape-description').innerText = description;

  // Set image (manual mapping since styles.json doesn't have image field)
  const imageMap = {
    apple: 'img_body/dangquatao.jpg',
    pear: 'img_body/dangquale.jpg',
    hourglass: 'img_body/dangdonghocat.jpg',
    rectangle: 'img_body/chunhat.png',
    'inverted-triangle': 'img_body/dangtamgiacnguoc.jpg',
  };
  const resultImage = document.getElementById('result-image');
  if (resultImage) {
    resultImage.src = imageMap[shapeData.title.toLowerCase().replace('dáng ', '').replace(' ', '-')] || '';
    resultImage.alt = title;
  }

  // Set recommendations
  const recommendationsList = document.getElementById('shape-recommendations');
  recommendationsList.innerHTML = '';
  recommendations.forEach(({ item, reason }) => {
    const li = document.createElement('li');
    li.innerHTML = `<i class="fas fa-check-circle"></i> ${item}<small>${reason}</small>`;
    recommendationsList.appendChild(li);
  });

  // Set avoid items
  const avoidList = document.getElementById('shape-avoid');
  avoidList.innerHTML = '';
  avoid.forEach(({ item, reason }) => {
    const li = document.createElement('li');
    li.innerHTML = `<i class="fas fa-times-circle"></i> ${item}<small>${reason}</small>`;
    avoidList.appendChild(li);
  });

  // Show result section with animation
  const resultSection = document.getElementById('result');
  resultSection.style.display = 'block';
  resultSection.style.animation = 'fadeIn 0.5s ease';

  // Scroll to result
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showShapeInfo(shape, gender) {
  displayResult({ ...window.bodyShapes[shape], gender });
}