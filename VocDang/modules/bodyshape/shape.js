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
    icon.addEventListener('click', function() {
      const shape = this.getAttribute('data-shape');
      showShapeInfo(shape);
    });
  });

  // Save result button
  document.getElementById('save-result')?.addEventListener('click', function() {
    const currentResult = localStorage.getItem('currentShapeResult');
    if (currentResult) {
      localStorage.setItem('savedShapeResult', currentResult);
      this.innerHTML = '<i class="fas fa-check"></i> Đã lưu';
      setTimeout(() => {
        this.innerHTML = '<i class="fas fa-bookmark"></i> Lưu kết quả';
      }, 2000);
    }
  });
});

function enableDarkMode() {
  document.body.classList.add('dark-mode');
  document.querySelector('.shape-analyzer').classList.add('dark-mode');
  document.querySelector('.navbar').classList.add('dark-mode');
}

const bodyShapes = {
  apple: {
    title: "Dáng Quả Táo",
    description: "Vóc dáng quả táo có đặc điểm phần thân trên (vai, ngực) rộng hơn hông, eo thường đầy đặn hơn. Bạn có đường cong tập trung ở phần trên cơ thể với chân thon gọn.",
    recommendations: [
      "Áo khoác dáng dài hoặc áo blazer để kéo dài thân hình",
      "Váy chữ A hoặc váy suông giúp che phần eo",
      "Quần ống rộng hoặc ống suông để cân bằng tỷ lệ cơ thể",
      "Áo cổ V sâu giúp kéo dài phần thân trên",
      "Đầm empire waist (đường eo cao ngay dưới ngực)"
    ],
    avoid: [
      "Áo bó sát hoặc áo crop top làm lộ phần eo",
      "Quần skinny sáng màu làm nổi bật phần dưới nhỏ hơn",
      "Áo có chi tiết rườm rà ở vùng vai hoặc ngực",
      "Thắt lưng rộng hoặc nổi bật ở eo",
      "Quần hoặc váy có túi lớn ở hông"
    ],
    image: "img_body/dangquatao.jpg"
  },
  pear: {
    title: "Dáng Quả Lê",
    description: "Vóc dáng quả lê có phần hông và đùi rộng hơn vai và ngực, tạo cảm giác phần dưới nặng hơn. Bạn có đường cong tập trung ở phần dưới cơ thể với thân trên thon gọn.",
    recommendations: [
      "Áo có vai phồng hoặc chi tiết nổi bật để thu hút sự chú ý lên thân trên",
      "Váy chữ A hoặc váy xòe giúp che phần hông rộng",
      "Quần tối màu, ống suông hoặc ống loe để cân bằng tỷ lệ",
      "Áo sáng màu hoặc họa tiết để làm nổi bật thân trên",
      "Đầm fit and flare (ôm phần thân trên và xòe từ eo)"
    ],
    avoid: [
      "Quần skinny sáng màu làm nổi bật phần hông",
      "Áo quá dài che mất vòng eo tự nhiên",
      "Váy bó sát làm lộ phần hông và đùi",
      "Quần có túi lớn hoặc chi tiết ở mông",
      "Áo cropped quá ngắn làm lộ phần hông rộng"
    ],
    image: "img_body/dangquale.jpg"
  },
  hourglass: {
    title: "Dáng Đồng Hồ Cát",
    description: "Vóc dáng đồng hồ cát có vai và hông cân đối, với vòng eo thon gọn rõ rệt. Bạn có đường cong cân đối và tỷ lệ cơ thể lý tưởng.",
    recommendations: [
      "Váy ôm hoặc váy bút chì tôn lên đường cong cơ thể",
      "Áo bó sát hoặc áo peplum làm nổi bật vòng eo",
      "Quần cạp cao hoặc thắt lưng để nhấn mạnh eo",
      "Đầm bodycon ôm sát cơ thể",
      "Áo crop top kết hợp với chân váy bút chì"
    ],
    avoid: [
      "Trang phục quá rộng làm mất đi đường cong tự nhiên",
      "Áo khoác dáng hộp hoặc quá dài che mất vòng eo",
      "Quần ống rộng quá mức làm mất cân đối",
      "Áo không có đường eo rõ ràng",
      "Trang phục nhiều lớp làm che khuất đường cong"
    ],
    image: "img_body/dangdonghocat.jpg"
  },
  rectangle: {
    title: "Dáng Chữ Nhật",
    description: "Vóc dáng chữ nhật có vai, eo và hông gần bằng nhau, tạo cảm giác thẳng và ít đường cong. Bạn có thân hình mảnh mai với ít sự khác biệt giữa các số đo.",
    recommendations: [
      "Áo có chi tiết bèo nhún hoặc peplum để tạo ảo giác vòng eo",
      "Váy xòe hoặc váy có thắt lưng để tạo đường cong",
      "Quần ống loe hoặc quần palazzo để thêm độ mềm mại",
      "Áo có họa tiết hoặc chi tiết để tạo chiều sâu",
      "Đầm wrap dress tạo đường cong nhân tạo"
    ],
    avoid: [
      "Trang phục quá bó sát làm lộ thiếu đường cong",
      "Áo dài thẳng không có điểm nhấn ở eo",
      "Quần ống suông không tạo được tỷ lệ cơ thể",
      "Trang phục đơn điệu một màu",
      "Áo tank top hoặc áo ba lỗ làm lộ vai thẳng"
    ],
    image: "img_body/chunhat.png"
  },
  invertedTriangle: {
    title: "Dáng Tam Giác Ngược",
    description: "Vai rộng hơn hông, thân trên nổi bật, hông nhỏ và chân thon. Cần phối đồ để cân bằng phần dưới.",
    recommendations: [
      "Chọn quần/váy sáng màu, xếp ly, ống rộng để tạo cân đối",
      "Váy chữ A, váy xòe, quần baggy hoặc ống loe",
      "Áo cổ chữ V, cổ sâu, màu tối, đơn giản",
      "Tránh áo cầu vai, tay phồng, áo cổ rộng"
    ],
    avoid: [
      "Áo có chi tiết nổi bật ở vai/ngực",
      "Áo cổ thuyền, áo sát nách",
      "Quần bó sát, váy bó sát phần dưới"
    ],
    image: "img_body/dangtamgiacnguoc.jpg"
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
  errorDiv.style.display = 'none';
  
  const inputs = [
    { value: shoulder, name: 'vai', min: 20, max: 100 },
    { value: chest, name: 'ngực', min: 50, max: 150 },
    { value: waist, name: 'eo', min: 40, max: 120 },
    { value: hip, name: 'hông', min: 50, max: 150 },
    { value: height, name: 'chiều cao', min: 100, max: 250 }
  ];

  for (const input of inputs) {
    if (isNaN(input.value) || input.value < input.min || input.value > input.max) {
      errorDiv.innerText = `Số đo ${input.name} phải từ ${input.min} đến ${input.max} cm.`;
      errorDiv.style.display = 'block';
      return;
    }
  }

  // Save inputs to localStorage
  const shapeData = { shoulder, chest, waist, hip, height };
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
    shape = 'invertedTriangle';
  } else {
    shape = 'rectangle';
  }

  // Display results
  displayResult(bodyShapes[shape]);
  
  // Save current result
  localStorage.setItem('currentShapeResult', JSON.stringify(bodyShapes[shape]));
});

function displayResult(shapeData) {
  document.getElementById('shape-title').innerText = shapeData.title;
  document.getElementById('shape-description').innerText = shapeData.description;
  
  // Set image
  const resultImage = document.getElementById('result-image');
  if (resultImage) {
    resultImage.src = shapeData.image;
    resultImage.alt = shapeData.title;
  }

  // Set recommendations
  const recommendationsList = document.getElementById('shape-recommendations');
  recommendationsList.innerHTML = '';
  shapeData.recommendations.forEach(item => {
    const li = document.createElement('li');
    li.innerHTML = `<i class="fas fa-check-circle"></i> ${item}`;
    recommendationsList.appendChild(li);
  });

  // Set avoid items
  const avoidList = document.getElementById('shape-avoid');
  avoidList.innerHTML = '';
  shapeData.avoid.forEach(item => {
    const li = document.createElement('li');
    li.innerHTML = `<i class="fas fa-times-circle"></i> ${item}`;
    avoidList.appendChild(li);
  });

  // Show result section with animation
  const resultSection = document.getElementById('result');
  resultSection.style.display = 'block';
  resultSection.style.animation = 'fadeIn 0.5s ease';
  
  // Scroll to result
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showShapeInfo(shape) {
  displayResult(bodyShapes[shape]);
}