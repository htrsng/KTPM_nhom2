document.addEventListener('DOMContentLoaded', () => {
    // Fade-in animation cho sản phẩm
    const products = document.querySelectorAll('.grid > div');
    products.forEach((product, index) => {
        product.style.opacity = '0';
        setTimeout(() => {
            product.style.transition = 'opacity 0.5s';
            product.style.opacity = '1';
        }, index * 100);
    });

    // Hamburger menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
});