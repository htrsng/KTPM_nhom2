// dathuong.js
document.addEventListener('DOMContentLoaded', () => {
  // Toggle menu trên thiết bị di động
  const menuToggle = document.getElementById('menu-toggle');
  const menu = document.getElementById('menu');

  menuToggle.addEventListener('click', () => {
    menu.classList.toggle('hidden');
  });

  // Danh sách sản phẩm mẫu
  const products = [
    {
      name: 'Sữa rửa mặt dịu nhẹ',
      price: '250.000 VNĐ',
      image: 'https://via.placeholder.com/300',
      description: 'Phù hợp cho da thường, làm sạch sâu.'
    },
    {
      name: 'Kem chống nắng SPF 50',
      price: '350.000 VNĐ',
      image: 'https://via.placeholder.com/300',
      description: 'Bảo vệ da khỏi tia UV hiệu quả.'
    },
    {
      name: 'Serum dưỡng ẩm',
      price: '450.000 VNĐ',
      image: 'https://via.placeholder.com/300',
      description: 'Cấp ẩm sâu, mang lại làn da mịn màng.'
    },
  ];

  // Hiển thị sản phẩm
  const productList = document.getElementById('product-list');
  products.forEach(product => {
    const productCard = document.createElement('div');
    productCard.className = 'product-card';
    productCard.innerHTML = `
      <img src="${product.image}" alt="${product.name}">
      <h3>${product.name}</h3>
      <p>${product.description}</p>
      <p class="font-bold text-pink-600">${product.price}</p>
      <button>Thêm vào giỏ hàng</button>
    `;
    productList.appendChild(productCard);
  });
});