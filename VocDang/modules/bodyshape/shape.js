document.addEventListener('DOMContentLoaded', function () {
  // Load saved data
  const savedData = JSON.parse(localStorage.getItem('shapeData'));
  if (savedData) {
    ['shoulder', 'chest', 'waist', 'hip', 'height', 'gender'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = savedData[id] || '';
    });
  }

  // Load saved result
  const savedResult = localStorage.getItem('savedShapeResult');
  if (savedResult) {
    displayResult(JSON.parse(savedResult));
  }


  const darkModeToggle = document.getElementById('dark-mode-toggle');
  const body = document.body;
  if (
    localStorage.getItem('darkMode') === 'enabled' ||
    (!localStorage.getItem('darkMode') && window.matchMedia('(prefers-color-scheme: dark)').matches)
  ) {
    body.classList.add('dark-mode');
    if (darkModeToggle) darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
  } else {
    body.classList.remove('dark-mode');
    if (darkModeToggle) darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
  }

  if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
      body.classList.toggle('dark-mode');
      if (body.classList.contains('dark-mode')) {
        localStorage.setItem('darkMode', 'enabled');
        darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
      } else {
        localStorage.setItem('darkMode', 'disabled');
        darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
      }
    });
  }


  window.bodyShapes = {
    apple: {
      title: "Dáng Quả Táo",
      description: "Vóc dáng quả táo đặc trưng bởi phần thân trên (vai và ngực) rộng hơn so với hông, vòng eo thường đầy đặn và ít thon gọn. Phần chân thường thon, tạo cảm giác phần trên cơ thể nặng hơn. Dáng này phổ biến ở cả nam và nữ, đặc biệt ở những người có xu hướng tích tụ mỡ ở vùng bụng.",
      recommendations_male: [
        {
          item: "Áo sơ mi dáng suông hoặc áo khoác dáng dài",
          reason: "Giúp kéo dài thân hình, che đi phần eo đầy đặn và tạo sự cân bằng giữa thân trên và thân dưới."
        },
        {
          item: "Quần jeans hoặc quần chinos ống suông, tối màu",
          reason: "Tối màu giúp thu gọn phần dưới, trong khi ống suông tạo tỷ lệ hài hòa với phần thân trên rộng."
        },
        {
          item: "Áo thun cổ chữ V hoặc áo polo không quá ôm",
          reason: "Cổ chữ V kéo dài thân trên, làm giảm sự chú ý vào vùng eo và tạo cảm giác thon gọn hơn."
        }
      ],
      recommendations_female: [
        {
          item: "Áo khoác dáng dài hoặc áo blazer cinched waist",
          reason: "Dáng dài giúp kéo dài thân hình, trong khi chi tiết cinched waist tạo ảo giác vòng eo thon hơn."
        },
        {
          item: "Váy chữ A hoặc váy suông",
          reason: "Che phần eo đầy đặn, tạo sự cân bằng bằng cách làm rộng nhẹ phần hông."
        },
        {
          item: "Quần ống rộng hoặc ống suông, cạp cao",
          reason: "Cạp cao nhấn vào phần eo, trong khi ống rộng cân bằng tỷ lệ giữa thân trên và thân dưới."
        }
      ],
      avoid_male: [
        {
          item: "Áo bó sát hoặc áo thun ôm",
          reason: "Làm lộ phần eo đầy đặn, khiến thân trên trông nặng nề hơn."
        },
        {
          item: "Quần skinny sáng màu",
          reason: "Làm nổi bật phần dưới nhỏ hơn, tạo sự mất cân đối với thân trên."
        },
        {
          item: "Áo sơ mi có họa tiết lớn ở vùng ngực hoặc vai",
          reason: "Thu hút sự chú ý vào phần thân trên đã rộng, làm mất cân đối tổng thể."
        }
      ],
      avoid_female: [
        {
          item: "Áo bó sát hoặc áo crop top",
          reason: "Làm lộ phần eo đầy đặn, khiến cơ thể trông kém thon gọn."
        },
        {
          item: "Quần skinny sáng màu",
          reason: "Tạo sự tương phản mạnh với thân trên, làm nổi bật sự mất cân đối."
        },
        {
          item: "Áo có chi tiết rườm rà ở vùng vai hoặc ngực",
          reason: "Làm phần thân trên trông rộng hơn, phá vỡ tỷ lệ cơ thể."
        }
      ]
    },
    pear: {
      title: "Dáng Quả Lê",
      description: "Vóc dáng quả lê có phần hông và đùi rộng hơn vai và ngực, tạo cảm giác phần dưới cơ thể nặng hơn. Phần thân trên thường thon gọn, vai nhỏ, và vòng eo thường rõ ràng. Dáng này thường thấy ở nữ nhiều hơn, nhưng nam giới cũng có thể có đặc điểm này.",
      recommendations_male: [
        {
          item: "Áo sơ mi hoặc áo thun có chi tiết ở vai (cầu vai, họa tiết)",
          reason: "Tạo cảm giác vai rộng hơn, cân bằng với phần hông rộng."
        },
        {
          item: "Quần jeans tối màu, ống suông hoặc hơi loe",
          reason: "Tối màu giúp thu gọn phần hông, trong khi ống suông/loe cân bằng tỷ lệ cơ thể."
        },
        {
          item: "Áo khoác dáng ngắn, vừa vặn",
          reason: "Nhấn mạnh phần thân trên, thu hút sự chú ý khỏi phần hông rộng."
        }
      ],
      recommendations_female: [
        {
          item: "Áo có vai phồng hoặc chi tiết nổi bật (bèo nhún, họa tiết)",
          reason: "Thu hút sự chú ý lên thân trên, làm vai trông rộng hơn để cân bằng với hông."
        },
        {
          item: "Váy chữ A hoặc váy xòe",
          reason: "Che phần hông rộng, tạo đường cong mềm mại và cân đối hơn."
        },
        {
          item: "Quần tối màu, ống suông hoặc ống loe",
          reason: "Giảm sự chú ý vào phần hông, đồng thời tạo tỷ lệ hài hòa với thân trên."
        }
      ],
      avoid_male: [
        {
          item: "Quần ống bó hoặc sáng màu",
          reason: "Làm nổi bật phần hông rộng, khiến cơ thể trông mất cân đối."
        },
        {
          item: "Áo quá dài che mất vòng eo",
          reason: "Che đi điểm mạnh là vòng eo thon, làm cơ thể trông thẳng hơn."
        },
        {
          item: "Áo sơ mi quá ôm ở phần thân dưới",
          reason: "Nhấn mạnh phần hông, làm mất cân đối với thân trên nhỏ hơn."
        }
      ],
      avoid_female: [
        {
          item: "Quần skinny sáng màu",
          reason: "Làm nổi bật phần hông và đùi, khiến phần dưới trông nặng nề hơn."
        },
        {
          item: "Áo quá dài che mất vòng eo tự nhiên",
          reason: "Làm mất điểm nhấn vòng eo, khiến cơ thể trông kém cân đối."
        },
        {
          item: "Váy bó sát",
          reason: "Làm lộ phần hông và đùi rộng, phá vỡ tỷ lệ cơ thể."
        }
      ]
    },
    hourglass: {
      title: "Dáng Đồng Hồ Cát",
      description: "Vóc dáng đồng hồ cát có vai và hông cân đối, với vòng eo thon gọn rõ rệt, tạo đường cong tự nhiên đẹp mắt. Đây là dáng người lý tưởng, thường được cả nam và nữ mong muốn, vì dễ dàng phù hợp với nhiều kiểu trang phục.",
      recommendations_male: [
        {
          item: "Áo sơ mi ôm vừa hoặc áo thun dáng fitted",
          reason: "Tôn lên vòng eo thon và đường cong tự nhiên của cơ thể."
        },
        {
          item: "Quần jeans cạp cao hoặc quần chinos ôm vừa",
          reason: "Nhấn mạnh tỷ lệ cân đối, làm nổi bật vòng eo và hông."
        },
        {
          item: "Áo khoác dáng ôm, ngắn (bomber hoặc blazer fitted)",
          reason: "Tôn lên đường cong cơ thể, giữ được sự cân đối giữa thân trên và dưới."
        }
      ],
      recommendations_female: [
        {
          item: "Váy ôm hoặc váy bút chì",
          reason: "Tôn lên đường cong tự nhiên, đặc biệt là vòng eo thon gọn."
        },
        {
          item: "Áo bó sát hoặc áo peplum",
          reason: "Làm nổi bật vòng eo, tạo điểm nhấn cho dáng đồng hồ cát."
        },
        {
          item: "Quần cạp cao hoặc thắt lưng",
          reason: "Nhấn mạnh vòng eo, giữ tỷ lệ cơ thể cân đối và hài hòa."
        }
      ],
      avoid_male: [
        {
          item: "Áo sơ mi hoặc áo khoác quá rộng",
          reason: "Che mất đường cong tự nhiên, làm cơ thể trông thẳng và kém nổi bật."
        },
        {
          item: "Quần ống rộng quá mức",
          reason: "Phá vỡ tỷ lệ cân đối, làm mất đi vẻ hài hòa của dáng đồng hồ cát."
        },
        {
          item: "Áo thun không có form dáng rõ ràng",
          reason: "Làm mất điểm nhấn vòng eo, khiến cơ thể trông kém nổi bật."
        }
      ],
      avoid_female: [
        {
          item: "Trang phục quá rộng",
          reason: "Che mất đường cong tự nhiên, làm cơ thể trông thẳng và kém nổi bật."
        },
        {
          item: "Áo khoác dáng hộp hoặc quá dài",
          reason: "Che mất vòng eo thon, làm mất đi đặc điểm nổi bật của dáng này."
        },
        {
          item: "Quần ống rộng quá mức",
          reason: "Phá vỡ tỷ lệ cân đối, làm cơ thể trông kém hài hòa."
        }
      ]
    },
    rectangle: {
      title: "Dáng Chữ Nhật",
      description: "Vóc dáng chữ nhật có vai, eo và hông gần bằng nhau, tạo cảm giác cơ thể thẳng và ít đường cong. Dáng này thường mang lại vẻ ngoài khỏe khoắn, phổ biến ở cả nam và nữ, đặc biệt ở những người có thân hình cân đối nhưng thiếu điểm nhấn ở eo.",
      recommendations_male: [
        {
          item: "Áo sơ mi có họa tiết hoặc chi tiết ở ngực (túi ngực, họa tiết lớn)",
          reason: "Tạo độ rộng cho thân trên, làm cơ thể trông có đường cong hơn."
        },
        {
          item: "Quần jeans ống loe hoặc quần chinos dáng thoải mái",
          reason: "Thêm độ mềm mại và tạo ảo giác đường cong ở phần hông."
        },
        {
          item: "Áo khoác có thắt lưng hoặc chi tiết ở eo",
          reason: "Tạo ảo giác vòng eo, giúp cơ thể trông bớt thẳng và có tỷ lệ hơn."
        }
      ],
      recommendations_female: [
        {
          item: "Áo có chi tiết bèo nhún hoặc peplum",
          reason: "Tạo ảo giác vòng eo thon gọn, thêm đường cong cho cơ thể."
        },
        {
          item: "Váy xòe hoặc váy có thắt lưng",
          reason: "Tạo điểm nhấn ở eo, làm cơ thể trông mềm mại và có đường cong hơn."
        },
        {
          item: "Quần ống loe hoặc quần palazzo",
          reason: "Thêm độ mềm mại ở phần dưới, tạo tỷ lệ hài hòa hơn cho cơ thể."
        }
      ],
      avoid_male: [
        {
          item: "Áo sơ mi hoặc áo thun quá bó",
          reason: "Làm lộ thân hình thẳng, khiến cơ thể trông thiếu đường cong."
        },
        {
          item: "Quần ống suông không có điểm nhấn",
          reason: "Không tạo được tỷ lệ, làm cơ thể trông thẳng hơn."
        },
        {
          item: "Áo khoác dáng dài, thẳng",
          reason: "Củng cố cảm giác thân hình thẳng, không tạo được điểm nhấn."
        }
      ],
      avoid_female: [
        {
          item: "Trang phục quá bó sát",
          reason: "Làm lộ sự thiếu đường cong, khiến cơ thể trông thẳng hơn."
        },
        {
          item: "Áo dài thẳng không có điểm nhấn ở eo",
          reason: "Không tạo được tỷ lệ, làm cơ thể trông thiếu điểm nhấn."
        },
        {
          item: "Quần ống suông không tạo được tỷ lệ",
          reason: "Làm cơ thể trông thẳng và kém mềm mại."
        }
      ]
    },
    "inverted-triangle": {
      title: "Dáng Tam Giác Ngược",
      description: "Vóc dáng tam giác ngược có vai rộng hơn hông, tạo cảm giác phần thân trên mạnh mẽ và phần dưới thon gọn hơn. Dáng này thường thấy ở nam giới có thân hình thể thao hoặc nữ giới có vai rộng, hông nhỏ, và vòng eo không quá rõ rệt.",
      recommendations_male: [
        {
          item: "Áo sơ mi dáng suông hoặc áo thun cổ tròn",
          reason: "Giảm độ rộng của vai, tạo cảm giác thân trên nhẹ nhàng hơn."
        },
        {
          item: "Quần jeans ống rộng hoặc quần chinos sáng màu",
          reason: "Làm rộng phần hông, cân bằng với vai rộng để tạo tỷ lệ hài hòa."
        },
        {
          item: "Áo khoác dáng dài, hơi ôm",
          reason: "Kéo dài thân hình, làm giảm sự chú ý vào vai rộng."
        }
      ],
      recommendations_female: [
        {
          item: "Áo cổ chữ V hoặc áo có chi tiết nhẹ nhàng (ren, bèo nhún)",
          reason: "Làm mềm phần vai, giảm cảm giác vai rộng và tạo sự nữ tính."
        },
        {
          item: "Váy chữ A hoặc váy xòe",
          reason: "Làm rộng phần hông, cân bằng với vai để tạo tỷ lệ hài hòa."
        },
        {
          item: "Quần ống loe hoặc palazzo",
          reason: "Thêm độ rộng cho phần dưới, giúp cân bằng với vai rộng."
        }
      ],
      avoid_male: [
        {
          item: "Áo thun hoặc sơ mi có cầu vai lớn",
          reason: "Làm vai trông rộng hơn, khiến cơ thể mất cân đối."
        },
        {
          item: "Quần skinny hoặc ống bó",
          reason: "Làm phần dưới trông nhỏ hơn, nhấn mạnh sự mất cân đối với vai."
        },
        {
          item: "Áo khoác ngắn, ôm sát",
          reason: "Làm lộ vai rộng, khiến thân trên trông nặng nề hơn."
        }
      ],
      avoid_female: [
        {
          item: "Áo có chi tiết phồng hoặc cầu vai",
          reason: "Làm vai trông rộng hơn, khiến cơ thể trông mất cân đối."
        },
        {
          item: "Quần skinny sáng màu",
          reason: "Làm nổi bật phần hông nhỏ, tạo sự tương phản mạnh với vai."
        },
        {
          item: "Áo crop top",
          reason: "Làm lộ sự mất cân đối giữa vai rộng và hông nhỏ."
        }
      ]
    }
  };
  window.bodyShapesLoaded = true;
  const loading = document.getElementById('loading');
  if (loading) loading.style.setProperty('display', 'none');

  // Shape icon click event
  document.querySelectorAll('.shape-icon').forEach(icon => {
    icon.addEventListener('click', function () {
      const shape = this.getAttribute('data-shape');
      const gender = document.getElementById('gender')?.value || 'female';
      if (window.bodyShapes?.[shape]) {
        showShapeInfo(shape, gender);
      }
    });
  });

  // Save result
  const saveResultBtn = document.getElementById('save-result');
  if (saveResultBtn) {
    saveResultBtn.addEventListener('click', function () {
      const currentResult = localStorage.getItem('currentShapeResult');
      if (currentResult) {
        localStorage.setItem('savedShapeResult', currentResult);
        this.innerHTML = '<i class="fas fa-check"></i> Đã lưu';
        setTimeout(() => {
          this.innerHTML = '<i class="fas fa-bookmark"></i> Lưu kết quả';
        }, 2000);
      }
    });
  }
});

// Form submit handler
document.getElementById('shape-form')?.addEventListener('submit', function (e) {
  e.preventDefault();

  const errorDiv = document.getElementById('error');
  if (!window.bodyShapesLoaded) {
    if (errorDiv) {
      errorDiv.innerText = 'Hệ thống đang tải dữ liệu. Vui lòng thử lại sau.';
      errorDiv.style.display = 'block';
    }
    return;
  }

  // Lấy dữ liệu từ form
  const shoulder = parseFloat(document.getElementById('shoulder')?.value);
  const chest = parseFloat(document.getElementById('chest')?.value);
  const waist = parseFloat(document.getElementById('waist')?.value);
  const hip = parseFloat(document.getElementById('hip')?.value);
  const height = parseFloat(document.getElementById('height')?.value);
  const gender = document.getElementById('gender')?.value;

  // Validate
  if (errorDiv) errorDiv.style.display = 'none';
  const inputs = [
    { value: shoulder, name: 'vai', min: 20, max: 100 },
    { value: chest, name: 'ngực', min: 50, max: 150 },
    { value: waist, name: 'eo', min: 40, max: 120 },
    { value: hip, name: 'hông', min: 50, max: 150 },
    { value: height, name: 'chiều cao', min: 100, max: 250 },
  ];

  for (const input of inputs) {
    if (isNaN(input.value) || input.value < input.min || input.value > input.max) {
      if (errorDiv) {
        errorDiv.innerText = `Số đo ${input.name} phải từ ${input.min} đến ${input.max} cm.`;
        errorDiv.style.display = 'block';
      }
      return;
    }
  }

  // Lưu localStorage
  const shapeData = { shoulder, chest, waist, hip, height, gender };
  localStorage.setItem('shapeData', JSON.stringify(shapeData));

  // Phân tích
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

  const resultData = { ...window.bodyShapes[shape], gender, shape }; // Thêm shape vào đây
  displayResult(resultData);
  localStorage.setItem('currentShapeResult', JSON.stringify(resultData));
});

// Hiển thị kết quả
function displayResult(shapeData) {
  const { title, description, gender } = shapeData;
  const recommendations = shapeData[`recommendations_${gender}`] || [];
  const avoid = shapeData[`avoid_${gender}`] || [];

  const shapeTitle = document.getElementById('shape-title');
  const shapeDescription = document.getElementById('shape-description');
  if (!shapeTitle || !shapeDescription) {
    console.error('Không tìm thấy shape-title hoặc shape-description trong DOM');
    return;
  }
  shapeTitle.textContent = title || 'Không có tiêu đề';
  shapeDescription.textContent = description || 'Không có mô tả';

  const imageMap = {
    apple: 'img_body/qt.png',
    pear: 'img_body/ql.png',
    hourglass: 'img_body/dhc.png',
    rectangle: 'img_body/hcn.png',
    'inverted-triangle': 'img_body/tgn.png',
  };

  const resultImage = document.getElementById('result-image');
  if (resultImage) {
    const imgKey = shapeData.shape || title?.toLowerCase().replace('dáng ', '').replace(/\s+/g, '-') || '';
    resultImage.src = imageMap[imgKey] || '';
    resultImage.alt = title || '';
    console.log('Image Key:', imgKey, 'Image Src:', resultImage.src); 
  } else {
    console.error('Không tìm thấy result-image trong DOM');
  }

  const recommendationsList = document.getElementById('shape-recommendations');
  const avoidList = document.getElementById('shape-avoid');
  if (recommendationsList && avoidList) {
    recommendationsList.innerHTML = '';
    recommendations.forEach(({ item, reason }) => {
      const li = document.createElement('li');
      li.innerHTML = `<i class="fas fa-check-circle"></i> ${item}<small>${reason}</small>`;
      recommendationsList.appendChild(li);
    });

    avoidList.innerHTML = '';
    avoid.forEach(({ item, reason }) => {
      const li = document.createElement('li');
      li.innerHTML = `<i class="fas fa-times-circle"></i> ${item}<small>${reason}</small>`;
      avoidList.appendChild(li);
    });
  } else {
    console.error('Không tìm thấy shape-recommendations hoặc shape-avoid trong DOM');
  }

  const resultSection = document.getElementById('result');
  if (resultSection) {
    resultSection.style.display = 'block';
    resultSection.style.animation = 'fadeIn 0.5s ease';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    console.error('Không tìm thấy result trong DOM');
  }
}

// Hiển thị từ icon
function showShapeInfo(shape, gender) {
  if (window.bodyShapes?.[shape]) {
    displayResult({ ...window.bodyShapes[shape], gender, shape });
  }
}

