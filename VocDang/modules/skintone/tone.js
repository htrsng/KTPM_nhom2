document.addEventListener('DOMContentLoaded', function () {
  // Load saved data from localStorage 
  const savedData = JSON.parse(localStorage.getItem('toneData'));
  if (savedData) {
    [
      { name: "vein_color", value: savedData.vein_color },
      { name: "clothing_color", value: savedData.clothing_color },
      { name: "jewelry", value: savedData.jewelry },
      { name: "sun_reaction", value: savedData.sun_reaction },
      { name: "hair_color", value: savedData.hair_color },
      { name: "eye_color", value: savedData.eye_color },
      { name: "skin_brightness", value: savedData.skin_brightness }
    ].forEach(item => {
      const input = document.querySelector(`input[name="${item.name}"][value="${item.value}"]`);
      if (input) input.checked = true;
    });
  }

  // Dark mode toggle
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  if (darkModeToggle) {
    if (localStorage.getItem('darkMode') === 'enabled') {
      document.body.classList.add('dark-mode');
      darkModeToggle.textContent = '☀️ Chế độ sáng';
    }
    darkModeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      if (document.body.classList.contains('dark-mode')) {
        localStorage.setItem('darkMode', 'enabled');
        darkModeToggle.textContent = '☀️ Chế độ sáng';
      } else {
        localStorage.setItem('darkMode', 'disabled');
        darkModeToggle.textContent = '🌙 Chế độ tối';
      }
    });
  }

  // Load palettes.json
  let palettes = null;
  fetch('../../data/palettes.json')
    .then(res => res.json())
    .then(data => {
      palettes = data;
    });

  // Form submit handler
  const toneForm = document.getElementById('tone-form');
  if (!toneForm) return;

  toneForm.addEventListener('submit', function (e) {
    e.preventDefault();

    // Lấy giá trị các trường
    const veinColor = document.querySelector('input[name="vein_color"]:checked')?.value;
    const clothingColor = document.querySelector('input[name="clothing_color"]:checked')?.value;
    const jewelry = document.querySelector('input[name="jewelry"]:checked')?.value;
    const sunReaction = document.querySelector('input[name="sun_reaction"]:checked')?.value;
    const hairColor = document.querySelector('input[name="hair_color"]:checked')?.value;
    const eyeColor = document.querySelector('input[name="eye_color"]:checked')?.value;
    const skinBrightness = document.querySelector('input[name="skin_brightness"]:checked')?.value;

    // Kiểm tra hợp lệ
    const errorDiv = document.getElementById('error');
    if (!veinColor || !clothingColor || !jewelry || !sunReaction || !hairColor || !eyeColor || !skinBrightness) {
      if (errorDiv) {
        errorDiv.innerText = 'Vui lòng trả lời tất cả các câu hỏi.';
        errorDiv.style.display = 'block';
      }
      return;
    }
    if (errorDiv) errorDiv.style.display = 'none';

    // Đảm bảo palettes đã load xong
    if (!palettes) {
      alert('Đang tải dữ liệu bảng màu, vui lòng thử lại sau!');
      return;
    }

    // Lưu lại lựa chọn
    const toneData = {
      vein_color: veinColor,
      clothing_color: clothingColor,
      jewelry,
      sun_reaction: sunReaction,
      hair_color: hairColor,
      eye_color: eyeColor,
      skin_brightness: skinBrightness
    };
    localStorage.setItem('toneData', JSON.stringify(toneData));

    // Xác định undertone
    const responses = [veinColor, clothingColor, jewelry, sunReaction, hairColor, eyeColor];
    const coolCount = responses.filter(r => r === 'cool').length;
    const warmCount = responses.filter(r => r === 'warm').length;

    let undertone;
    if (coolCount >= 4) undertone = 'cool';
    else if (warmCount >= 4) undertone = 'warm';
    else undertone = 'neutral';

    // Xác định mùa
    let season;
    if (undertone === 'cool') {
      season = skinBrightness === 'bright' ? 'Winter' : 'Summer';
    } else if (undertone === 'warm') {
      season = skinBrightness === 'bright' ? 'Spring' : 'Autumn';
    } else {
      season = skinBrightness === 'bright' ? 'Summer' : 'Autumn';
    }

    // Lấy dữ liệu từ palettes.json
    const selectedPalette = palettes[season];
    const toneTitle = document.getElementById('tone-title');
    const toneDesc = document.getElementById('tone-description');
    const paletteDiv = document.getElementById('color-palette');
    const styleList = document.getElementById('style-suggestions');
    const lipList = document.getElementById('lip-suggestions');
    const hairList = document.getElementById('hair-suggestions');
    const resultDiv = document.getElementById('result');

    if (toneTitle) toneTitle.innerText = selectedPalette.title || '';
    if (toneDesc) toneDesc.innerText = `Tông da: ${undertone === 'cool' ? 'Lạnh' : undertone === 'warm' ? 'Ấm' : 'Trung tính'}. ${selectedPalette.description || ''}`;

    if (paletteDiv) {
      paletteDiv.innerHTML = '';
      (selectedPalette.colors || []).forEach(color => {
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch';
        swatch.style.backgroundColor = color.hex || color;
        swatch.setAttribute('data-color', color.name || color.hex || color);
        paletteDiv.appendChild(swatch);
      });
      addColorTooltips(); // Thêm tooltip cho màu sắc
    }

    if (styleList) {
      styleList.innerHTML = '';
      (selectedPalette.style_suggestions || []).forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        styleList.appendChild(li);
      });
    }

    if (lipList) {
      lipList.innerHTML = '';
      (selectedPalette.lip_colors || []).forEach(item => {
        const li = document.createElement('li');
        li.innerText = item.name || item;
        lipList.appendChild(li);
      });
    }

    if (hairList) {
      hairList.innerHTML = '';
      (selectedPalette.hair_colors || []).forEach(item => {
        const li = document.createElement('li');
        li.innerText = item.name || item;
        hairList.appendChild(li);
      });
    }

    if (resultDiv) {
      resultDiv.style.display = 'block';
      window.scrollTo({
        top: resultDiv.offsetTop - 100,
        behavior: 'smooth'
      });
    }
  });

  // Thêm tooltip cho các màu sắc
  function addColorTooltips() {
    document.querySelectorAll('.color-swatch').forEach(swatch => {
      const colorName = swatch.getAttribute('data-color');
      const tooltip = document.createElement('span');
      tooltip.className = 'tooltip';
      tooltip.textContent = colorName;
      swatch.appendChild(tooltip);
    });
  }
});
