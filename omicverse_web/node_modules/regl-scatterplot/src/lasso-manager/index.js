import {
  assign,
  identity,
  l2Norm,
  l2PointDist,
  nextAnimationFrame,
  pipe,
  throttleAndDebounce,
  withConstructor,
  withStaticProperty,
} from '@flekschas/utils';

import {
  DEFAULT_BRUSH_SIZE,
  DEFAULT_LASSO_MIN_DELAY,
  DEFAULT_LASSO_MIN_DIST,
  DEFAULT_LASSO_START_INITIATOR_SHOW,
  DEFAULT_LASSO_TYPE,
  LASSO_HIDE_START_INITIATOR_TIME,
  LASSO_SHOW_START_INITIATOR_TIME,
} from './constants.js';

import {
  DEFAULT_LASSO_LONG_PRESS_AFTER_EFFECT_TIME,
  DEFAULT_LASSO_LONG_PRESS_EFFECT_DELAY,
  DEFAULT_LASSO_LONG_PRESS_REVERT_EFFECT_TIME,
  DEFAULT_LASSO_LONG_PRESS_TIME,
} from '../constants.js';

import {
  createLongPressInAnimations,
  createLongPressOutAnimations,
} from './create-long-press-animations.js';
import createLongPressElements from './create-long-press-elements.js';
import { exponentialMovingAverage } from './utils.js';

const ifNotNull = (v, alternative = null) => (v === null ? alternative : v);

let cachedLassoStylesheets;

const getLassoStylesheets = () => {
  if (!cachedLassoStylesheets) {
    const lassoStyleEl = document.createElement('style');
    document.head.appendChild(lassoStyleEl);
    cachedLassoStylesheets = lassoStyleEl.sheet;
  }
  return cachedLassoStylesheets;
};

const addRule = (rule) => {
  const lassoStylesheets = getLassoStylesheets();
  const currentNumRules = lassoStylesheets.rules.length;
  lassoStylesheets.insertRule(rule, currentNumRules);
  return currentNumRules;
};

const removeRule = (index) => {
  getLassoStylesheets().deleteRule(index);
};

const inAnimation = `${LASSO_SHOW_START_INITIATOR_TIME}ms ease scaleInFadeOut 0s 1 normal backwards`;

const createInAnimationRule = (opacity, scale, rotate) => `
@keyframes scaleInFadeOut {
  0% {
    opacity: ${opacity};
    transform: translate(-50%,-50%) scale(${scale}) rotate(${rotate}deg);
  }
  10% {
    opacity: 1;
    transform: translate(-50%,-50%) scale(1) rotate(${rotate + 20}deg);
  }
  100% {
    opacity: 0;
    transform: translate(-50%,-50%) scale(0.9) rotate(${rotate + 60}deg);
  }
}
`;
let inAnimationRuleIndex = null;

const outAnimation = `${LASSO_HIDE_START_INITIATOR_TIME}ms ease fadeScaleOut 0s 1 normal backwards`;

const createOutAnimationRule = (opacity, scale, rotate) => `
@keyframes fadeScaleOut {
  0% {
    opacity: ${opacity};
    transform: translate(-50%,-50%) scale(${scale}) rotate(${rotate}deg);
  }
  100% {
    opacity: 0;
    transform: translate(-50%,-50%) scale(0) rotate(${rotate}deg);
  }
}
`;
let outAnimationRuleIndex = null;

export const createLasso = (
  element,
  {
    onDraw: initialOnDraw = identity,
    onStart: initialOnStart = identity,
    onEnd: initialOnEnd = identity,
    enableInitiator:
      initialenableInitiator = DEFAULT_LASSO_START_INITIATOR_SHOW,
    initiatorParentElement: initialInitiatorParentElement = document.body,
    longPressIndicatorParentElement:
      initialLongPressIndicatorParentElement = document.body,
    minDelay: initialMinDelay = DEFAULT_LASSO_MIN_DELAY,
    minDist: initialMinDist = DEFAULT_LASSO_MIN_DIST,
    pointNorm: initialPointNorm = identity,
    type: initialType = DEFAULT_LASSO_TYPE,
    brushSize: initialBrushSize = DEFAULT_BRUSH_SIZE,
  } = {},
) => {
  let enableInitiator = initialenableInitiator;
  let initiatorParentElement = initialInitiatorParentElement;
  let longPressIndicatorParentElement = initialLongPressIndicatorParentElement;

  let onDraw = initialOnDraw;
  let onStart = initialOnStart;
  let onEnd = initialOnEnd;

  let minDelay = initialMinDelay;
  let minDist = initialMinDist;

  let pointNorm = initialPointNorm;

  let type = initialType;
  let brushSize = initialBrushSize;

  const initiator = document.createElement('div');
  const initiatorId =
    Math.random().toString(36).substring(2, 5) +
    Math.random().toString(36).substring(2, 5);
  initiator.id = `lasso-initiator-${initiatorId}`;
  initiator.style.position = 'fixed';
  initiator.style.display = 'flex';
  initiator.style.justifyContent = 'center';
  initiator.style.alignItems = 'center';
  initiator.style.zIndex = 99;
  initiator.style.width = '4rem';
  initiator.style.height = '4rem';
  initiator.style.borderRadius = '4rem';
  initiator.style.opacity = 0.5;
  initiator.style.transform = 'translate(-50%,-50%) scale(0) rotate(0deg)';

  const {
    longPress,
    longPressCircle,
    longPressCircleLeft,
    longPressCircleRight,
    longPressEffect,
  } = createLongPressElements();

  let isMouseDown = false;
  let isLasso = false;
  let lassoPos = [];
  let lassoPosFlat = [];
  let lassoBrushCenterPos = [];
  let lassoBrushNormals = [];
  let prevMousePos;
  let longPressIsStarting = false;

  let longPressMainInAnimationRuleIndex = null;
  let longPressEffectInAnimationRuleIndex = null;
  let longPressCircleLeftInAnimationRuleIndex = null;
  let longPressCircleRightInAnimationRuleIndex = null;
  let longPressCircleInAnimationRuleIndex = null;
  let longPressMainOutAnimationRuleIndex = null;
  let longPressEffectOutAnimationRuleIndex = null;
  let longPressCircleLeftOutAnimationRuleIndex = null;
  let longPressCircleRightOutAnimationRuleIndex = null;
  let longPressCircleOutAnimationRuleIndex = null;

  const mouseUpHandler = () => {
    isMouseDown = false;
  };

  const getMousePosition = (event) => {
    const { left, top } = element.getBoundingClientRect();

    return [event.clientX - left, event.clientY - top];
  };

  window.addEventListener('mouseup', mouseUpHandler);

  const resetInitiatorStyle = () => {
    initiator.style.opacity = 0.5;
    initiator.style.transform = 'translate(-50%,-50%) scale(0) rotate(0deg)';
  };

  const getCurrentTransformStyle = (node, hasRotated) => {
    const computedStyle = getComputedStyle(node);
    const opacity = +computedStyle.opacity;
    // The css rule `transform: translate(-1, -1) scale(0.5);` is represented as
    // `matrix(0.5, 0, 0, 0.5, -1, -1)`
    const m = computedStyle.transform.match(/([0-9.-]+)+/g);

    const a = +m[0];
    const b = +m[1];

    const scale = Math.sqrt(a * a + b * b);
    let rotate = Math.atan2(b, a) * (180 / Math.PI);

    rotate = hasRotated && rotate <= 0 ? 360 + rotate : rotate;

    return { opacity, scale, rotate };
  };

  const showInitiator = (event) => {
    if (!enableInitiator || isMouseDown) {
      return;
    }

    const x = event.clientX;
    const y = event.clientY;
    initiator.style.top = `${y}px`;
    initiator.style.left = `${x}px`;

    const style = getCurrentTransformStyle(initiator);
    const opacity = style.opacity;
    const scale = style.scale;
    const rotate = style.rotate;
    initiator.style.opacity = opacity;
    initiator.style.transform = `translate(-50%,-50%) scale(${scale}) rotate(${rotate}deg)`;

    initiator.style.animation = 'none';

    // See https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Animations/Tips
    // why we need to wait for two animation frames
    nextAnimationFrame().then(() => {
      if (inAnimationRuleIndex !== null) {
        removeRule(inAnimationRuleIndex);
      }

      inAnimationRuleIndex = addRule(
        createInAnimationRule(opacity, scale, rotate),
      );

      initiator.style.animation = inAnimation;

      nextAnimationFrame().then(() => {
        resetInitiatorStyle();
      });
    });
  };

  const hideInitiator = () => {
    const { opacity, scale, rotate } = getCurrentTransformStyle(initiator);
    initiator.style.opacity = opacity;
    initiator.style.transform = `translate(-50%,-50%) scale(${scale}) rotate(${rotate}deg)`;

    initiator.style.animation = 'none';

    nextAnimationFrame(2).then(() => {
      if (outAnimationRuleIndex !== null) {
        removeRule(outAnimationRuleIndex);
      }

      outAnimationRuleIndex = addRule(
        createOutAnimationRule(opacity, scale, rotate),
      );

      initiator.style.animation = outAnimation;

      nextAnimationFrame().then(() => {
        resetInitiatorStyle();
      });
    });
  };

  const showLongPressIndicator = (
    x,
    y,
    {
      time = DEFAULT_LASSO_LONG_PRESS_TIME,
      extraTime = DEFAULT_LASSO_LONG_PRESS_AFTER_EFFECT_TIME,
      delay = DEFAULT_LASSO_LONG_PRESS_EFFECT_DELAY,
    } = {
      time: DEFAULT_LASSO_LONG_PRESS_TIME,
      extraTime: DEFAULT_LASSO_LONG_PRESS_AFTER_EFFECT_TIME,
      delay: DEFAULT_LASSO_LONG_PRESS_EFFECT_DELAY,
    },
  ) => {
    longPressIsStarting = true;

    const mainStyle = getComputedStyle(longPress);
    longPress.style.color = mainStyle.color;
    longPress.style.top = `${y}px`;
    longPress.style.left = `${x}px`;
    longPress.style.animation = 'none';

    const circleStyle = getComputedStyle(longPressCircle);
    longPressCircle.style.clipPath = circleStyle.clipPath;
    longPressCircle.style.opacity = circleStyle.opacity;
    longPressCircle.style.animation = 'none';

    const effectStyle = getCurrentTransformStyle(longPressEffect);
    longPressEffect.style.opacity = effectStyle.opacity;
    longPressEffect.style.transform = `scale(${effectStyle.scale})`;
    longPressEffect.style.animation = 'none';

    const circleLeftStyle = getCurrentTransformStyle(longPressCircleLeft);
    longPressCircleLeft.style.transform = `rotate(${circleLeftStyle.rotate}deg)`;
    longPressCircleLeft.style.animation = 'none';

    const circleRightStyle = getCurrentTransformStyle(longPressCircleRight);
    longPressCircleRight.style.transform = `rotate(${circleRightStyle.rotate}deg)`;
    longPressCircleRight.style.animation = 'none';

    nextAnimationFrame().then(() => {
      if (!longPressIsStarting) {
        return;
      }

      if (longPressCircleInAnimationRuleIndex !== null) {
        removeRule(longPressCircleInAnimationRuleIndex);
      }
      if (longPressCircleRightInAnimationRuleIndex !== null) {
        removeRule(longPressCircleRightInAnimationRuleIndex);
      }
      if (longPressCircleLeftInAnimationRuleIndex !== null) {
        removeRule(longPressCircleLeftInAnimationRuleIndex);
      }
      if (longPressEffectInAnimationRuleIndex !== null) {
        removeRule(longPressEffectInAnimationRuleIndex);
      }
      if (longPressMainInAnimationRuleIndex !== null) {
        removeRule(longPressMainInAnimationRuleIndex);
      }

      const { rules, names } = createLongPressInAnimations({
        time,
        extraTime,
        delay,
        currentColor: mainStyle.color || 'currentcolor',
        targetColor: longPress.dataset.activeColor,
        effectOpacity: effectStyle.opacity || 0,
        effectScale: effectStyle.scale || 0,
        circleLeftRotation: circleLeftStyle.rotate || 0,
        circleRightRotation: circleRightStyle.rotate || 0,
        circleClipPath: circleStyle.clipPath || 'inset(0 0 0 50%)',
        circleOpacity: circleStyle.opacity || 0,
      });

      longPressMainInAnimationRuleIndex = addRule(rules.main);
      longPressEffectInAnimationRuleIndex = addRule(rules.effect);
      longPressCircleLeftInAnimationRuleIndex = addRule(rules.circleLeft);
      longPressCircleRightInAnimationRuleIndex = addRule(rules.circleRight);
      longPressCircleInAnimationRuleIndex = addRule(rules.circle);

      longPress.style.animation = names.main;
      longPressEffect.style.animation = names.effect;
      longPressCircleLeft.style.animation = names.circleLeft;
      longPressCircleRight.style.animation = names.circleRight;
      longPressCircle.style.animation = names.circle;
    });
  };

  const hideLongPressIndicator = (
    { time = DEFAULT_LASSO_LONG_PRESS_REVERT_EFFECT_TIME } = {
      time: DEFAULT_LASSO_LONG_PRESS_REVERT_EFFECT_TIME,
    },
  ) => {
    if (!longPressIsStarting) {
      return;
    }

    longPressIsStarting = false;

    const mainStyle = getComputedStyle(longPress);
    longPress.style.color = mainStyle.color;
    longPress.style.animation = 'none';

    const circleStyle = getComputedStyle(longPressCircle);
    longPressCircle.style.clipPath = circleStyle.clipPath;
    longPressCircle.style.opacity = circleStyle.opacity;
    longPressCircle.style.animation = 'none';

    const effectStyle = getCurrentTransformStyle(longPressEffect);
    longPressEffect.style.opacity = effectStyle.opacity;
    longPressEffect.style.transform = `scale(${effectStyle.scale})`;
    longPressEffect.style.animation = 'none';

    // The first half of the circle animation, the clip-path is set to `inset(0px 0px 0px 50%)`.
    // In the second half it's set to `inset(0px)`. Hence we can look at the second to last
    // character to determine if the animatation has progressed passed half time.
    const isAnimatedMoreThan50Percent =
      circleStyle.clipPath.slice(-2, -1) === 'x';

    const circleLeftStyle = getCurrentTransformStyle(
      longPressCircleLeft,
      isAnimatedMoreThan50Percent,
    );
    longPressCircleLeft.style.transform = `rotate(${circleLeftStyle.rotate}deg)`;
    longPressCircleLeft.style.animation = 'none';

    const circleRightStyle = getCurrentTransformStyle(longPressCircleRight);
    longPressCircleRight.style.transform = `rotate(${circleRightStyle.rotate}deg)`;
    longPressCircleRight.style.animation = 'none';

    nextAnimationFrame().then(() => {
      if (longPressCircleOutAnimationRuleIndex !== null) {
        removeRule(longPressCircleOutAnimationRuleIndex);
      }
      if (longPressCircleRightOutAnimationRuleIndex !== null) {
        removeRule(longPressCircleRightOutAnimationRuleIndex);
      }
      if (longPressCircleLeftOutAnimationRuleIndex !== null) {
        removeRule(longPressCircleLeftOutAnimationRuleIndex);
      }
      if (longPressEffectOutAnimationRuleIndex !== null) {
        removeRule(longPressEffectOutAnimationRuleIndex);
      }
      if (longPressMainOutAnimationRuleIndex !== null) {
        removeRule(longPressMainOutAnimationRuleIndex);
      }

      const { rules, names } = createLongPressOutAnimations({
        time,
        currentColor: mainStyle.color || 'currentcolor',
        targetColor: longPress.dataset.color,
        effectOpacity: effectStyle.opacity || 0,
        effectScale: effectStyle.scale || 0,
        circleLeftRotation: circleLeftStyle.rotate || 0,
        circleRightRotation: circleRightStyle.rotate || 0,
        circleClipPath: circleStyle.clipPath || 'inset(0px)',
        circleOpacity: circleStyle.opacity || 1,
      });

      longPressMainOutAnimationRuleIndex = addRule(rules.main);
      longPressEffectOutAnimationRuleIndex = addRule(rules.effect);
      longPressCircleLeftOutAnimationRuleIndex = addRule(rules.circleLeft);
      longPressCircleRightOutAnimationRuleIndex = addRule(rules.circleRight);
      longPressCircleOutAnimationRuleIndex = addRule(rules.circle);

      longPress.style.animation = names.main;
      longPressEffect.style.animation = names.effect;
      longPressCircleLeft.style.animation = names.circleLeft;
      longPressCircleRight.style.animation = names.circleRight;
      longPressCircle.style.animation = names.circle;
    });
  };

  const draw = () => {
    onDraw(lassoPos, lassoPosFlat);
  };

  const extendFreeform = (point) => {
    lassoPos.push(point);
    lassoPosFlat.push(point[0], point[1]);
  };

  const extendRectangle = (point) => {
    const [x, y] = point;
    const [startX, startY] = lassoPos[0];

    lassoPos[1] = [x, startY];
    lassoPos[2] = [x, y];
    lassoPos[3] = [startX, y];
    lassoPos[4] = [startX, startY];

    lassoPosFlat[2] = x;
    lassoPosFlat[3] = startY;
    lassoPosFlat[4] = x;
    lassoPosFlat[5] = y;
    lassoPosFlat[6] = startX;
    lassoPosFlat[7] = y;
    lassoPosFlat[8] = startX;
    lassoPosFlat[9] = startY;
  };

  const startBrush = (point) => {
    lassoBrushCenterPos.push(point);
  };

  const getNormalizedBrushSize = () =>
    Math.abs(pointNorm([0, 0])[0] - pointNorm([brushSize / 2, 0])[0]);

  const getBrushNormal = (point1, point2, w) => {
    const [x1, y1] = point1;
    const [x2, y2] = point2;

    const dx = x1 - x2;
    const dy = y1 - y2;
    const dn = l2Norm([dx, dy]);

    return [(+dy / dn) * w, (-dx / dn) * w];
  };

  const extendBrush = (point) => {
    const prevPoint = lassoBrushCenterPos.at(-1);

    const width = getNormalizedBrushSize();
    let [nx, ny] = getBrushNormal(point, prevPoint, width);

    const N = lassoBrushCenterPos.length;

    if (N === 1) {
      // In this special case, we have to add the initial two points and normal
      // because when the first brush point was set the direction is undefined.
      const pl = [prevPoint[0] + nx, prevPoint[1] + ny];
      const pr = [prevPoint[0] - nx, prevPoint[1] - ny];

      lassoPos.push(pl, pr);
      lassoPosFlat.push(pl[0], pl[1], pr[0], pr[1]);
      lassoBrushNormals.push([nx, ny]);
    } else {
      // In this case, we have to adjust the previous normal to create a proper
      // line join by taking the middle between the current and previous normal.
      // const prevPrevPoint = lassoBrushCenterPos.at(-2);
      [nx, ny] = getBrushNormal(point, prevPoint, width);

      const nextRawBrushNormals = [...lassoBrushNormals, [nx, ny]];

      // However, to avoid jittery lines we're smoothing the normal
      [nx, ny] = exponentialMovingAverage(nextRawBrushNormals, 1, 10);

      const [pnx, pny] = lassoBrushNormals.at(-1);

      const pnx2 = (nx + pnx) / 2;
      const pny2 = (ny + pny) / 2;

      const pl = [prevPoint[0] + pnx2, prevPoint[1] + pny2];
      const pr = [prevPoint[0] - pnx2, prevPoint[1] - pny2];

      // We're going to replace the previous left and right points
      lassoPos.splice(N - 1, 2, pl, pr);
      lassoPosFlat.splice(2 * (N - 1), 4, pl[0], pl[1], pr[0], pr[1]);
      lassoBrushNormals.splice(N, 1, [pnx2, pny2]);
    }

    const pl = [point[0] + nx, point[1] + ny];
    const pr = [point[0] - nx, point[1] - ny];

    lassoPos.splice(N, 0, pl, pr);
    lassoPosFlat.splice(2 * N, 0, pl[0], pl[1], pr[0], pr[1]);

    lassoBrushCenterPos.push(point);
    lassoBrushNormals.push([nx, ny]);
  };

  let extendLasso = extendFreeform;
  let startLasso = extendFreeform;

  const extend = (currMousePos) => {
    if (prevMousePos) {
      const d = l2PointDist(
        currMousePos[0],
        currMousePos[1],
        prevMousePos[0],
        prevMousePos[1],
      );

      if (d > minDist) {
        prevMousePos = currMousePos;

        extendLasso(pointNorm(currMousePos));

        if (lassoPos.length > 1) {
          draw();
        }
      }
    } else {
      if (!isLasso) {
        isLasso = true;
        onStart();
      }
      prevMousePos = currMousePos;
      const point = pointNorm(currMousePos);
      startLasso(point);
    }
  };

  const extendDb = throttleAndDebounce(extend, minDelay, minDelay);

  const extendPublic = (event, debounced) => {
    const mousePosition = getMousePosition(event);
    if (debounced) {
      return extendDb(mousePosition);
    }
    return extend(mousePosition);
  };

  const clear = () => {
    lassoPos = [];
    lassoPosFlat = [];
    lassoBrushCenterPos = [];
    lassoBrushNormals = [];
    prevMousePos = undefined;
    draw();
  };

  const initiatorClickHandler = (event) => {
    showInitiator(event);
  };

  const initiatorMouseDownHandler = () => {
    isMouseDown = true;
    isLasso = true;
    clear();
    onStart();
  };

  const initiatorMouseLeaveHandler = () => {
    hideInitiator();
  };

  const end = ({ merge = false, remove = false } = {}) => {
    isLasso = false;

    const currLassoPos = [...lassoPos];
    const currLassoPosFlat = [...lassoPosFlat];

    extendDb.cancel();

    clear();

    // When `currLassoPos` is empty the user didn't actually lasso
    if (currLassoPos.length > 0) {
      onEnd(currLassoPos, currLassoPosFlat, { merge, remove });
    }

    return currLassoPos;
  };

  const setExtendLasso = (newType) => {
    switch (newType) {
      case 'rectangle': {
        type = newType;
        extendLasso = extendRectangle;
        // This is on purpose. The start of a rectangle & freeform are the same
        startLasso = extendFreeform;
        break;
      }

      case 'brush': {
        type = newType;
        extendLasso = extendBrush;
        startLasso = startBrush;
        break;
      }

      default: {
        type = 'freeform';
        extendLasso = extendFreeform;
        startLasso = extendFreeform;
        break;
      }
    }
  };

  const get = (property) => {
    if (property === 'onDraw') {
      return onDraw;
    }
    if (property === 'onStart') {
      return onStart;
    }
    if (property === 'onEnd') {
      return onEnd;
    }
    if (property === 'enableInitiator') {
      return enableInitiator;
    }
    if (property === 'minDelay') {
      return minDelay;
    }
    if (property === 'minDist') {
      return minDist;
    }
    if (property === 'pointNorm') {
      return pointNorm;
    }
    if (property === 'type') {
      return type;
    }
    if (property === 'brushSize') {
      return brushSize;
    }
  };

  const set = ({
    onDraw: newOnDraw = null,
    onStart: newOnStart = null,
    onEnd: newOnEnd = null,
    enableInitiator: newEnableInitiator = null,
    initiatorParentElement: newInitiatorParentElement = null,
    longPressIndicatorParentElement: newLongPressIndicatorParentElement = null,
    minDelay: newMinDelay = null,
    minDist: newMinDist = null,
    pointNorm: newPointNorm = null,
    type: newType = null,
    brushSize: newBrushSize = null,
  } = {}) => {
    onDraw = ifNotNull(newOnDraw, onDraw);
    onStart = ifNotNull(newOnStart, onStart);
    onEnd = ifNotNull(newOnEnd, onEnd);
    enableInitiator = ifNotNull(newEnableInitiator, enableInitiator);
    minDelay = ifNotNull(newMinDelay, minDelay);
    minDist = ifNotNull(newMinDist, minDist);
    pointNorm = ifNotNull(newPointNorm, pointNorm);
    brushSize = ifNotNull(newBrushSize, brushSize);

    if (
      newInitiatorParentElement !== null &&
      newInitiatorParentElement !== initiatorParentElement
    ) {
      initiatorParentElement.removeChild(initiator);
      newInitiatorParentElement.appendChild(initiator);
      initiatorParentElement = newInitiatorParentElement;
    }

    if (
      newLongPressIndicatorParentElement !== null &&
      newLongPressIndicatorParentElement !== longPressIndicatorParentElement
    ) {
      longPressIndicatorParentElement.removeChild(longPress);
      newLongPressIndicatorParentElement.appendChild(longPress);
      longPressIndicatorParentElement = newLongPressIndicatorParentElement;
    }

    if (enableInitiator) {
      initiator.addEventListener('click', initiatorClickHandler);
      initiator.addEventListener('mousedown', initiatorMouseDownHandler);
      initiator.addEventListener('mouseleave', initiatorMouseLeaveHandler);
    } else {
      initiator.removeEventListener('mousedown', initiatorMouseDownHandler);
      initiator.removeEventListener('mouseleave', initiatorMouseLeaveHandler);
    }

    if (newType !== null) {
      setExtendLasso(newType);
    }
  };

  const destroy = () => {
    initiatorParentElement.removeChild(initiator);
    longPressIndicatorParentElement.removeChild(longPress);
    window.removeEventListener('mouseup', mouseUpHandler);
    initiator.removeEventListener('click', initiatorClickHandler);
    initiator.removeEventListener('mousedown', initiatorMouseDownHandler);
    initiator.removeEventListener('mouseleave', initiatorMouseLeaveHandler);
  };

  const withPublicMethods = () => (self) =>
    assign(self, {
      clear,
      destroy,
      end,
      extend: extendPublic,
      get,
      set,
      showInitiator,
      hideInitiator,
      showLongPressIndicator,
      hideLongPressIndicator,
    });

  initiatorParentElement.appendChild(initiator);
  longPressIndicatorParentElement.appendChild(longPress);

  set({
    onDraw,
    onStart,
    onEnd,
    enableInitiator,
    initiatorParentElement,
    type,
    brushSize,
  });

  return pipe(
    withStaticProperty('initiator', initiator),
    withStaticProperty('longPressIndicator', longPress),
    withPublicMethods(),
    withConstructor(createLasso),
  )({});
};

export default createLasso;
