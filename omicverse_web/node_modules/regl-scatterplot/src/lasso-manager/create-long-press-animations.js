import {
  DEFAULT_LASSO_LONG_PRESS_AFTER_EFFECT_TIME,
  DEFAULT_LASSO_LONG_PRESS_EFFECT_DELAY,
  DEFAULT_LASSO_LONG_PRESS_REVERT_EFFECT_TIME,
  DEFAULT_LASSO_LONG_PRESS_TIME,
} from '../constants.js';

const getInTime = (p, time, extraTime) => (1 - p) * time + extraTime;

const getMainInAnimation = (t, d) =>
  `${t}ms ease-out mainIn ${d}ms 1 normal forwards`;

const getEffectInAnimation = (t, d) =>
  `${t}ms ease-out effectIn ${d}ms 1 normal forwards`;

const getCircleLeftInAnimation = (t, d) =>
  `${t}ms linear leftSpinIn ${d}ms 1 normal forwards`;

const getCircleRightInAnimation = (t, d) =>
  `${t}ms linear rightSpinIn ${d}ms 1 normal forwards`;

const getCircleInAnimation = (t, d) =>
  `${t}ms linear circleIn ${d}ms 1 normal forwards`;

const getMainIn = (mainEffectPercent, currentColor, targetColor) => `
  @keyframes mainIn {
    0% {
      color: ${currentColor};
      opacity: 0;
    }
    0%, ${mainEffectPercent}% {
      color: ${currentColor};
      opacity: 1;
    }
    100% {
      color: ${targetColor};
      opacity: 0.8;
    }
  }
`;

const getEffectIn = (mainEffectPercent, afterEffectPercent, opacity, scale) => `
  @keyframes effectIn {
    0%, ${mainEffectPercent}% {
      opacity: ${opacity};
      transform: scale(${scale});
    }
    ${afterEffectPercent}% {
      opacity: 0.66;
      transform: scale(1.5);
    }
    99% {
      opacity: 0;
      transform: scale(2);
    }
    100% {
      opacity: 0;
      transform: scale(0);
    }
  }
`;

const getCircleIn = (halfMainEffectPercent, clipPath, opacity) => `
  @keyframes circleIn {
    0% {
      clip-path: ${clipPath};
      opacity: ${opacity};
    }
    ${halfMainEffectPercent}% {
      clip-path: ${clipPath};
      opacity: 1;
    }
    ${halfMainEffectPercent + 0.01}%, 100% {
      clip-path: inset(0);
      opacity: 1;
    }
  }
`;

const getCircleLeftIn = (mainEffectPercent, angle) => `
  @keyframes leftSpinIn {
    0% {
      transform: rotate(${angle}deg);
    }
    ${mainEffectPercent}%, 100% {
      transform: rotate(360deg);
    }
  }
`;

const getCircleRightIn = (halfMainEffectPercent, angle) => `
  @keyframes rightSpinIn {
    0% {
      transform: rotate(${angle}deg);
    }
    ${halfMainEffectPercent}%, 100% {
      transform: rotate(180deg);
    }
  }
`;

export const createLongPressInAnimations = ({
  time = DEFAULT_LASSO_LONG_PRESS_TIME,
  extraTime = DEFAULT_LASSO_LONG_PRESS_AFTER_EFFECT_TIME,
  delay = DEFAULT_LASSO_LONG_PRESS_EFFECT_DELAY,
  currentColor,
  targetColor,
  effectOpacity,
  effectScale,
  circleLeftRotation,
  circleRightRotation,
  circleClipPath,
  circleOpacity,
}) => {
  const p = circleLeftRotation / 360;
  const actualTime = getInTime(p, time, extraTime);
  const longPressPercent = Math.round((((1 - p) * time) / actualTime) * 100);
  const halfLongPressPercent = Math.round(longPressPercent / 2);
  const afterEffectPercent = longPressPercent + (100 - longPressPercent) / 4;

  return {
    rules: {
      main: getMainIn(longPressPercent, currentColor, targetColor),
      effect: getEffectIn(
        longPressPercent,
        afterEffectPercent,
        effectOpacity,
        effectScale,
      ),
      circleRight: getCircleRightIn(halfLongPressPercent, circleRightRotation),
      circleLeft: getCircleLeftIn(longPressPercent, circleLeftRotation),
      circle: getCircleIn(halfLongPressPercent, circleClipPath, circleOpacity),
    },
    names: {
      main: getMainInAnimation(actualTime, delay),
      effect: getEffectInAnimation(actualTime, delay),
      circleLeft: getCircleLeftInAnimation(actualTime, delay),
      circleRight: getCircleRightInAnimation(actualTime, delay),
      circle: getCircleInAnimation(actualTime, delay),
    },
  };
};

const getMainOutAnimation = (t) => `${t}ms linear mainOut 0s 1 normal forwards`;

const getEffectOutAnimation = (t) =>
  `${t}ms linear effectOut 0s 1 normal forwards`;

const getCircleLeftOutAnimation = (t) =>
  `${t}ms linear leftSpinOut 0s 1 normal forwards`;

const getCircleRightOutAnimation = (t) =>
  `${t}ms linear rightSpinOut 0s 1 normal forwards`;

const getCircleOutAnimation = (t) =>
  `${t}ms linear circleOut 0s 1 normal forwards`;

const getMainOut = (currentColor, targetColor) => `
  @keyframes mainOut {
    0% {
      color: ${currentColor};
    }
    100% {
      color: ${targetColor};
    }
  }
`;

const getEffectOut = (opacity, scale) => `
  @keyframes effectOut {
    0% {
      opacity: ${opacity};
      transform: scale(${scale});
    }
    99% {
      opacity: 0;
      transform: scale(${scale + 0.5});
    }
    100% {
      opacity: 0;
      transform: scale(0);
    }
  }
`;

const getCircleRightOut = (halfEffectPercent, angle) => `
  @keyframes rightSpinOut {
    0%, ${halfEffectPercent}% {
      transform: rotate(${angle}deg);
    }
    100% {
      transform: rotate(0deg);
    }
`;

const getCircleLeftOut = (angle) => `
  @keyframes leftSpinOut {
    0% {
      transform: rotate(${angle}deg);
    }
    100% {
      transform: rotate(0deg);
    }
  }
`;

const getCircleOut = (halfEffectPercent, clipPath, opacity) => `
  @keyframes circleOut {
    0%, ${halfEffectPercent}% {
      clip-path: ${clipPath};
      opacity: ${opacity};
    }
    ${halfEffectPercent + 0.01}% {
      clip-path: inset(0 0 0 50%);
      opacity: ${opacity};
    }
    100% {
      clip-path: inset(0 0 0 50%);
      opacity: 0;
    }
  }
`;

export const createLongPressOutAnimations = ({
  time = DEFAULT_LASSO_LONG_PRESS_REVERT_EFFECT_TIME,
  currentColor,
  targetColor,
  effectOpacity,
  effectScale,
  circleLeftRotation,
  circleRightRotation,
  circleClipPath,
  circleOpacity,
}) => {
  const p = circleLeftRotation / 360;
  const actualTime = p * time;
  const rotatedPercent = Math.min(100, p * 100);
  const halfPercent =
    rotatedPercent > 50 ? Math.round((1 - 50 / rotatedPercent) * 100) : 0;

  return {
    rules: {
      main: getMainOut(currentColor, targetColor),
      effect: getEffectOut(effectOpacity, effectScale),
      circleRight: getCircleRightOut(halfPercent, circleRightRotation),
      circleLeft: getCircleLeftOut(circleLeftRotation),
      circle: getCircleOut(halfPercent, circleClipPath, circleOpacity),
    },
    names: {
      main: getMainOutAnimation(actualTime),
      effect: getEffectOutAnimation(actualTime),
      circleRight: getCircleLeftOutAnimation(actualTime),
      circleLeft: getCircleRightOutAnimation(actualTime),
      circle: getCircleOutAnimation(actualTime),
    },
  };
};
