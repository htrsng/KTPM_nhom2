document.addEventListener('DOMContentLoaded', function() {
  // Load saved data from localStorage
  const savedData = JSON.parse(localStorage.getItem('toneData'));
  if (savedData) {
    document.querySelector(`input[name="vein_color"][value="${savedData.vein_color}"]`).checked = true;
    document.querySelector(`input[name="clothing_color"][value="${savedData.clothing_color}"]`).checked = true;
    document.querySelector(`input[name="jewelry"][value="${savedData.jewelry}"]`).checked = true;
    document.querySelector(`input[name="sun_reaction"][value="${savedData.sun_reaction}"]`).checked = true;
    document.querySelector(`input[name="hair_color"][value="${savedData.hair_color}"]`).checked = true;
    document.querySelector(`input[name="eye_color"][value="${savedData.eye_color}"]`).checked = true;
    document.querySelector(`input[name="skin_brightness"][value="${savedData.skin_brightness}"]`).checked = true;
  }

  // Dark mode toggle
  const darkModeToggle = document.getElementById('dark-mode-toggle');
  if (localStorage.getItem('darkMode') === 'enabled') {
    document.body.classList.add('dark-mode');
    document.querySelector('.tone-analyzer').classList.add('dark-mode');
    document.querySelector('#result').classList.add('dark-mode');
    document.querySelector('.result-details').classList.add('dark-mode');
    document.querySelector('.navbar').classList.add('dark-mode');
    document.querySelector('footer').classList.add('dark-mode');
    document.querySelectorAll('.question').forEach(q => q.classList.add('dark-mode'));
    document.querySelectorAll('.options').forEach(opt => opt.classList.add('dark-mode'));
    document.querySelectorAll('.color-swatch').forEach(swatch => swatch.classList.add('dark-mode'));
    darkModeToggle.textContent = '☀️ Chế độ sáng';
  }

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    document.querySelector('.tone-analyzer').classList.toggle('dark-mode');
    document.querySelector('#result').classList.toggle('dark-mode');
    document.querySelector('.result-details').classList.toggle('dark-mode');
    document.querySelector('.navbar').classList.toggle('dark-mode');
    document.querySelector('footer').classList.toggle('dark-mode');
    document.querySelectorAll('.question').forEach(q => q.classList.toggle('dark-mode'));
    document.querySelectorAll('.options').forEach(opt => opt.classList.toggle('dark-mode'));
    document.querySelectorAll('.color-swatch').forEach(swatch => swatch.classList.toggle('dark-mode'));
    if (document.body.classList.contains('dark-mode')) {
      localStorage.setItem('darkMode', 'enabled');
      darkModeToggle.textContent = '☀️ Chế độ sáng';
    } else {
      localStorage.setItem('darkMode', 'disabled');
      darkModeToggle.textContent = '🌙 Chế độ tối';
    }
  });
});

document.getElementById('tone-form').addEventListener('submit', function(e) {
  e.preventDefault();

  // Get form inputs
  const veinColor = document.querySelector('input[name="vein_color"]:checked').value;
  const clothingColor = document.querySelector('input[name="clothing_color"]:checked').value;
  const jewelry = document.querySelector('input[name="jewelry"]:checked').value;
  const sunReaction = document.querySelector('input[name="sun_reaction"]:checked').value;
  const hairColor = document.querySelector('input[name="hair_color"]:checked').value;
  const eyeColor = document.querySelector('input[name="eye_color"]:checked').value;
  const skinBrightness = document.querySelector('input[name="skin_brightness"]:checked').value;

  // Input validation
  const errorDiv = document.getElementById('error');
  if (!veinColor || !clothingColor || !jewelry || !sunReaction || !hairColor || !eyeColor || !skinBrightness) {
    errorDiv.innerText = 'Vui lòng trả lời tất cả các câu hỏi.';
    errorDiv.style.display = 'block';
    return;
  }
  errorDiv.style.display = 'none';

  // Save inputs to localStorage
  const toneData = { vein_color: veinColor, clothing_color: clothingColor, jewelry, sun_reaction: sunReaction, hair_color: hairColor, eye_color: eyeColor, skin_brightness: skinBrightness };
  localStorage.setItem('toneData', JSON.stringify(toneData));

  // Calculate undertone
  const responses = [veinColor, clothingColor, jewelry, sunReaction, hairColor, eyeColor];
  const coolCount = responses.filter(r => r === 'cool').length;
  const warmCount = responses.filter(r => r === 'warm').length;
  const neutralCount = responses.filter(r => r === 'neutral').length;

  let undertone;
  if (coolCount >= 4) {
    undertone = 'cool';
  } else if (warmCount >= 4) {
    undertone = 'warm';
  } else {
    undertone = 'neutral';
  }

  // Determine seasonal palette
  let season;
  if (undertone === 'cool') {
    season = skinBrightness === 'bright' ? 'Winter' : 'Summer';
  } else if (undertone === 'warm') {
    season = skinBrightness === 'bright' ? 'Spring' : 'Autumn';
  } else {
    // For neutral, default to Summer or Autumn based on brightness
    season = skinBrightness === 'bright' ? 'Summer' : 'Autumn';
  }

  // Mock data (replace with fetch from palettes.json in production)
  const palettes = {
    Spring: {
      title: "Mùa Xuân",
      description: "Tông màu ấm, tươi sáng, phù hợp với mùa xuân. Da sáng với undertone ấm, rạng rỡ với các màu nhẹ nhàng.",
      colors: ["#FF6F61", "#FFD700", "#90EE90", "#87CEEB"],
      style_suggestions: ["Váy hoa nhẹ nhàng", "Áo màu pastel", "Phụ kiện vàng sáng"],
      lip_colors: ["Hồng đào", "San hô", "Đỏ cam nhẹ"],
      hair_colors: ["Nâu vàng", "Vàng mật ong", "Nâu caramel"]
    },
    Summer: {
      title: "Mùa Hè",
      description: "Tông màu lạnh, dịu nhẹ, phù hợp với mùa hè. Da trung bình với undertone lạnh, hợp với màu nhạt và nhẹ.",
      colors: ["#4682B4", "#D3D3D3", "#FFB6C1", "#E6E6FA"],
      style_suggestions: ["Áo sơ mi trắng", "Quần jeans xanh nhạt", "Phụ kiện bạc"],
      lip_colors: ["Hồng phấn", "Tím nhạt", "Đỏ berry"],
      hair_colors: ["Nâu tro", "Vàng bạch kim", "Đen xanh"]
    },
    Autumn: {
      title: "Mùa Thu",
      description: "Tông màu ấm, trầm, phù hợp với mùa thu. Da trung bình hoặc ngăm với undertone ấm, hợp với màu đất.",
      colors: ["#8B4513", "#DAA520", "#228B22", "#A0522D"],
      style_suggestions: ["Áo len nâu", "Khăn quàng màu đất", "Bốt da"],
      lip_colors: ["Đỏ gạch", "Cam cháy", "Nâu đất"],
      hair_colors: ["Nâu đỏ", "Nâu hạt dẻ", "Đồng ánh đỏ"]
    },
    Winter: {
      title: "Mùa Đông",
      description: "Tông màu lạnh, đậm, phù hợp với mùa đông. Da sáng với undertone lạnh, hợp với màu sắc nổi bật.",
      colors: ["#000080", "#4B0082", "#DC143C", "#000000"],
      style_suggestions: ["Áo khoác đen", "Váy màu jewel tone", "Phụ kiện ánh kim"],
      lip_colors: ["Đỏ đậm", "Tím đậm", "Fuchsia"],
      hair_colors: ["Đen tuyền", "Nâu lạnh", "Xám bạc"]
    }
  };

  // Display results
  const selectedPalette = palettes[season];
  document.getElementById('tone-title').innerText = selectedPalette.title;
  document.getElementById('tone-description').innerText = `Tông da: ${undertone === 'cool' ? 'Lạnh' : undertone === 'warm' ? 'Ấm' : 'Trung tính'}. ${selectedPalette.description}`;

  const paletteDiv = document.getElementById('color-palette');
  paletteDiv.innerHTML = '';
  selectedPalette.colors.forEach(color => {
    const swatch = document.createElement('div');
    swatch.className = 'color-swatch';
    swatch.style.backgroundColor = color;
    if (document.body.classList.contains('dark-mode')) {
      swatch.classList.add('dark-mode');
    }
    paletteDiv.appendChild(swatch);
  });

  const styleList = document.getElementById('style-suggestions');
  styleList.innerHTML = '';
  selectedPalette.style_suggestions.forEach(item => {
    const li = document.createElement('li');
    li.innerText = item;
    styleList.appendChild(li);
  });

  const lipList = document.getElementById('lip-suggestions');
  lipList.innerHTML = '';
  selectedPalette.lip_colors.forEach(item => {
    const li = document.createElement('li');
    li.innerText = item;
    lipList.appendChild(li);
  });

  const hairList = document.getElementById('hair-suggestions');
  hairList.innerHTML = '';
  selectedPalette.hair_colors.forEach(item => {
    const li = document.createElement('li');
    li.innerText = item;
    hairList.appendChild(li);
  });

  document.getElementById('result').style.display = 'block';
});