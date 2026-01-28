const documentButtons = document.querySelectorAll(".mkapi-document-toggle");

documentButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const element = button.parentElement.parentElement.nextElementSibling;
    let isInvisible = element.style.display === "none";
    element.style.display = isInvisible ? "block" : "none";
    const icon = button.querySelector("i");
    if (icon) {
      icon.className = isInvisible
        ? "fa-regular fa-square-minus"
        : "fa-regular fa-square-plus";
    }
  });
});

const sectionButtons = document.querySelectorAll(".mkapi-section-toggle");

sectionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const element = button.parentElement.parentElement.nextElementSibling;
    let isInvisible = element.style.display === "none";
    element.style.display = isInvisible ? "block" : "none";
    const icon = button.querySelector("i");
    if (icon) {
      icon.className = isInvisible
        ? "fa-regular fa-square-minus"
        : "fa-regular fa-square-plus";
    }
  });
});

const parentButtons = document.querySelectorAll(".mkapi-parent-toggle");

parentButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const elements = document.querySelectorAll(".mkapi-object-parent");
    let isVisible = elements[0].style.display === "inline";

    elements.forEach((element) => {
      element.style.display = isVisible ? "none" : "inline";
    });

    const buttons = document.querySelectorAll(".mkapi-parent-toggle");
    buttons.forEach((button) => {
      const icon = button.querySelector("i");
      if (icon) {
        icon.className = isVisible
          ? "fa-solid fa-square-plus"
          : "fa-solid fa-square-xmark";
      }
    });
  });
});
