export const createLongPressElements = () => {
  const longPress = document.createElement('div');
  const longPressId =
    Math.random().toString(36).substring(2, 5) +
    Math.random().toString(36).substring(2, 5);
  longPress.id = `lasso-long-press-${longPressId}`;
  longPress.style.position = 'fixed';
  longPress.style.width = '1.25rem';
  longPress.style.height = '1.25rem';
  longPress.style.pointerEvents = 'none';
  longPress.style.transform = 'translate(-50%,-50%)';

  const longPressCircle = document.createElement('div');
  longPressCircle.style.position = 'absolute';
  longPressCircle.style.top = 0;
  longPressCircle.style.left = 0;
  longPressCircle.style.width = '1.25rem';
  longPressCircle.style.height = '1.25rem';
  longPressCircle.style.clipPath = 'inset(0px 0px 0px 50%)';
  longPressCircle.style.opacity = 0;
  longPress.appendChild(longPressCircle);

  const longPressCircleLeft = document.createElement('div');
  longPressCircleLeft.style.position = 'absolute';
  longPressCircleLeft.style.top = 0;
  longPressCircleLeft.style.left = 0;
  longPressCircleLeft.style.width = '0.8rem';
  longPressCircleLeft.style.height = '0.8rem';
  longPressCircleLeft.style.border = '0.2rem solid currentcolor';
  longPressCircleLeft.style.borderRadius = '0.8rem';
  longPressCircleLeft.style.clipPath = 'inset(0px 50% 0px 0px)';
  longPressCircleLeft.style.transform = 'rotate(0deg)';
  longPressCircle.appendChild(longPressCircleLeft);

  const longPressCircleRight = document.createElement('div');
  longPressCircleRight.style.position = 'absolute';
  longPressCircleRight.style.top = 0;
  longPressCircleRight.style.left = 0;
  longPressCircleRight.style.width = '0.8rem';
  longPressCircleRight.style.height = '0.8rem';
  longPressCircleRight.style.border = '0.2rem solid currentcolor';
  longPressCircleRight.style.borderRadius = '0.8rem';
  longPressCircleRight.style.clipPath = 'inset(0px 50% 0px 0px)';
  longPressCircleRight.style.transform = 'rotate(0deg)';
  longPressCircle.appendChild(longPressCircleRight);

  const longPressEffect = document.createElement('div');
  longPressEffect.style.position = 'absolute';
  longPressEffect.style.top = 0;
  longPressEffect.style.left = 0;
  longPressEffect.style.width = '1.25rem';
  longPressEffect.style.height = '1.25rem';
  longPressEffect.style.borderRadius = '1.25rem';
  longPressEffect.style.background = 'currentcolor';
  longPressEffect.style.transform = 'scale(0)';
  longPressEffect.style.opacity = 0;
  longPress.appendChild(longPressEffect);

  return {
    longPress,
    longPressCircle,
    longPressCircleLeft,
    longPressCircleRight,
    longPressEffect,
  };
};

export default createLongPressElements;
