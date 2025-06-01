document.addEventListener('DOMContentLoaded', function() {
  // Load saved data from localStorage
  const savedData = JSON.parse(localStorage.getItem('shapeData'));
  if (savedData) {
    document.getElementById('shoulder').value = savedData.shoulder || '';
    document.getElementById('chest').value = savedData.chest || '';
    document.getElementById('waist').value = savedData.waist || '';
    document.getElementById('hip').value = savedData.hip || '';
    document.getElementById('height').value = savedData.height || '';
  }

  // Dark mode toggle
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  if (localStorage.getItem('darkMode') === 'enabled') {
    document.body.classList.add('dark-mode');
    document.querySelector('.shape-analyzer').classList.add('dark-mode');
    document.querySelector('#result').classList.add('dark-mode');
    document.querySelector('.result-details').classList.add('dark-mode');
    document.querySelector('.navbar').classList.add('dark-mode');
    document.querySelector('footer').classList.add('dark-mode');
    darkModeToggle.textContent = '☀️ Chế độ sáng';
  }

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.querySelector('.shape-analyzer').classList.toggle('dark-mode');
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

// Mock data (to be replaced with fetch from styles.json)
const bodyShapes = {
  apple: {
    title: "Dáng Quả Táo",
    description: "Vóc dáng quả táo có đặc điểm phần thân trên (vai, ngực) rộng hơn hông, eo thường đầy đặn hơn.",
    recommendations: [
      "Áo khoác dáng dài hoặc áo blazer để kéo dài thân hình.",
      "Váy chữ A hoặc váy suông giúp che phần eo.",
      "Quần ống rộng hoặc ống suông để cân bằng tỷ lệ cơ thể."
    ],
    avoid: [
      "Áo bó sát hoặc áo crop top làm lộ phần eo.",
      "Quần skinny sáng màu làm nổi bật phần dưới nhỏ hơn.",
      "Áo có chi tiết rườm rà ở vùng vai hoặc ngực."
    ]
  },
  pear: {
    title: "Dáng Quả Lê",
    description: "Vóc dáng quả lê có phần hông và đùi rộng hơn vai và ngực, tạo cảm giác phần dưới nặng hơn.",
    recommendations: [
      "Áo có vai phồng hoặc chi tiết nổi bật để thu hút sự chú ý lên thân trên.",
      "Váy chữ A hoặc váy xòe giúp che phần hông rộng.",
      "Quần tối màu, ống suông hoặc ống loe để cân bằng tỷ lệ."
    ],
    avoid: [
      "Quần skinny sáng màu làm nổi bật phần hông.",
      "Áo quá dài che mất vòng eo tự nhiên.",
      "Váy bó sát làm lộ phần hông và đùi."
    ]
  },
  hourglass: {
    title: "Dáng Đồng Hồ Cát",
    description: "Vóc dáng đồng hồ cát có vai và hông cân đối, với vòng eo thon gọn rõ rệt.",
    recommendations: [
      "Váy ôm hoặc váy bút chì tôn lên đường cong cơ thể.",
      "Áo bó sát hoặc áo peplum làm nổi bật vòng eo.",
      "Quần cạp cao hoặc thắt lưng để nhấn mạnh eo."
    ],
    avoid: [
      "Trang phục quá rộng làm mất đi đường cong tự nhiên.",
      "Áo khoác dáng hộp hoặc quá dài che mất vòng eo.",
      "Quần ống rộng quá mức làm mất cân đối."
    ]
  },
  rectangle: {
    title: "Dáng Chữ Nhật",
    description: "Vóc dáng chữ nhật có vai, eo và hông gần bằng nhau, tạo cảm giác thẳng và ít đường cong.",
    recommendations: [
      "Áo có chi tiết bèo nhún hoặc peplum để tạo ảo giác vòng eo.",
      "Váy xòe hoặc váy có thắt lưng để tạo đường cong.",
      "Quần ống loe hoặc quần palazzo để thêm độ mềm mại."
    ],
    avoid: [
      "Trang phục quá bó sát làm lộ thiếu đường cong.",
      "Áo dài thẳng không có điểm nhấn ở eo.",
      "Quần ống suông không tạo được tỷ lệ cơ thể."
    ]
  }
};

document.getElementById('shape-form').addEventListener('submit', function(e) {
  e.preventDefault();

  // Get form inputs
  const shoulder = parseFloat(document.getElementById('shoulder').value);
  const chest = parseFloat(document.getElementById('chest').value);
  const waist = parseFloat(document.getElementById('waist').value);
  const hip = parseFloat(document.getElementById('hip').value);
  const height = parseFloat(document.getElementById('height').value);

  // Input validation
  const errorDiv = document.getElementById('error');
  if (shoulder < 20 || shoulder > 100) {
    errorDiv.innerText = 'Số đo vai phải từ 20 đến 100 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  if (chest < 50 || chest > 150) {
    errorDiv.innerText = 'Số đo ngực phải từ 50 đến 150 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  if (waist < 40 || waist > 120) {
    errorDiv.innerText = 'Số đo eo phải từ 40 đến 120 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  if (hip < 50 || hip > 150) {
    errorDiv.innerText = 'Số đo hông phải từ 50 đến 150 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  if (height < 100 || height > 250) {
    errorDiv.innerText = 'Chiều cao phải từ 100 đến 250 cm.';
    errorDiv.style.display = 'block';
    return;
  }
  errorDiv.style.display = 'none';

  // Save inputs to localStorage
  const shapeData = { shoulder, chest, waist, hip, height };
  localStorage.setItem('shapeData', JSON.stringify(shapeData));

  // Calculate body shape based on ratios
  const shoulderToHipRatio = shoulder / hip;
  const waistToHipRatio = waist / hip;
  const waistToShoulderRatio = waist / shoulder;

  let shape;
  if (shoulderToHipRatio >= 1.05 && waistToHipRatio > 0.85) {
    shape = 'apple'; // Upper body wider, fuller waist
  } else if (shoulderToHipRatio < 0.95 && waistToHipRatio > 0.8) {
    shape = 'pear'; // Lower body wider
  } else if (shoulderToHipRatio >= 0.95 && shoulderToHipRatio <= 1.05 && waistToHipRatio <= 0.75) {
    shape = 'hourglass'; // Balanced shoulders/hips, defined waist
  } else {
    shape = 'rectangle'; // Similar measurements, less defined waist
  }

  // Display results
  const selectedShape = bodyShapes[shape];
  document.getElementById('shape-title').innerText = selectedShape.title;
  document.getElementById('shape-description').innerText = selectedShape.description;

  const recommendationsList = document.getElementById('shape-recommendations');
  recommendationsList.innerHTML = '';
  selectedShape.recommendations.forEach(item => {
    const li = document.createElement('li');
    li.innerText = item;
    recommendationsList.appendChild(li);
  });

  const avoidList = document.getElementById('shape-avoid');
  avoidList.innerHTML = '';
  selectedShape.avoid.forEach(item => {
    const li = document.createElement('li');
    li.innerText = item;
    avoidList.appendChild(li);
  });

  document.getElementById('result').style.display = 'block';
});