// Convert hex to RGB
function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Set CSS variables for RGB colors
function setColorVariables() {
    const root = document.documentElement;
    const primaryRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--primary').trim());
    root.style.setProperty('--primary-rgb', `${primaryRgb.r}, ${primaryRgb.g}, ${primaryRgb.b}`);
    
    const successRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--success').trim());
    root.style.setProperty('--success-rgb', `${successRgb.r}, ${successRgb.g}, ${successRgb.b}`);
    
    const warningRgb = hexToRgb(getComputedStyle(root).getPropertyValue('--warning').trim());
    root.style.setProperty('--warning-rgb', `${warningRgb.r}, ${warningRgb.g}, ${warningRgb.b}`);
}

// Theme management
document.addEventListener('DOMContentLoaded', function() {
    setColorVariables();
    
    // Theme toggle functionality
    const themeToggleBtn = document.querySelector('.theme-toggle');
    if (themeToggleBtn) {
        // Set initial theme based on system preference
        const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme');
        const initialTheme = savedTheme || (prefersDarkMode ? 'night' : 'day');
        
        if (initialTheme === 'night') {
            document.documentElement.classList.add('night-mode');
            themeToggleBtn.innerHTML = '<i>☀️</i> Light Mode';
        } else {
            themeToggleBtn.innerHTML = '<i>🌙</i> Dark Mode';
        }
        
        themeToggleBtn.addEventListener('click', function() {
            const isNightMode = document.documentElement.classList.toggle('night-mode');
            localStorage.setItem('theme', isNightMode ? 'night' : 'day');
            themeToggleBtn.innerHTML = isNightMode ? '<i>☀️</i> Light Mode' : '<i>🌙</i> Dark Mode';
            
            // Switch plot images
            document.querySelectorAll('.plot-img.day-mode').forEach(img => {
                img.style.display = isNightMode ? 'none' : 'block';
            });
            document.querySelectorAll('.plot-img.night-mode').forEach(img => {
                img.style.display = isNightMode ? 'block' : 'none';
            });
        });
    }
    
    // Navigation functionality
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Update active link
                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
    
    // Apply fade-in animations
    document.querySelectorAll('.card').forEach((el, index) => {
        el.style.animationDelay = `${index * 0.1}s`;
        el.classList.add('fade-in');
    });
}); 