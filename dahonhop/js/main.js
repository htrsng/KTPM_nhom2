
document.addEventListener('DOMContentLoaded', () => {
    // Gán --index cho các product-card và nav-item để hỗ trợ stagger animation
    const products = document.querySelectorAll('.grid > div');
    const navItems = document.querySelectorAll('.nav-links .nav-item');
    
    products.forEach((product, index) => {
        product.style.setProperty('--index', index); 
    });
    
    navItems.forEach((item, index) => {
        item.style.setProperty('--index', index); 
    });

    // Intersection Observer để kích hoạt stagger animation khi product-card trong viewport
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible'); 
                observer.unobserve(entry.target); 
            }
        });
    }, observerOptions);

    products.forEach(product => {
        product.style.opacity = '0'; 
        observer.observe(product); 
    });

    // Hamburger menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        hamburger.classList.toggle('open'); 
    });

    // Đóng menu khi click vào nav-item
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navLinks.classList.remove('active');
            hamburger.classList.remove('open');
        });
    });

    // Đóng menu khi click bên ngoài
    document.addEventListener('click', (event) => {
        if (!navLinks.contains(event.target) && !hamburger.contains(event.target)) {
            navLinks.classList.remove('active');
            hamburger.classList.remove('open');
        }
    });

    // Thêm hiệu ứng scroll nhẹ cho nav-item khi hover
    navItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.transition = 'transform 0.2s ease';
            item.style.transform = 'translateY(-2px)';
        });
        item.addEventListener('mouseleave', () => {
            item.style.transform = 'translateY(0)';
        });
    });
});
