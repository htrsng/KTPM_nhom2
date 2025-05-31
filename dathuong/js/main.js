document.addEventListener('DOMContentLoaded', () => {
    const products = document.querySelectorAll('.grid > div');
    products.forEach((product, index) => {
        product.style.opacity = '0';
        setTimeout(() => {
            product.style.transition = 'opacity 0.5s';
            product.style.opacity = '1';
        }, index * 100);
    });
});