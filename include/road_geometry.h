#pragma once

#include <cmath>
#include "raymath.h"

namespace CarGame {
namespace RoadGeometry {

    constexpr float CURVE_AMPLITUDE = 15.0f;
    constexpr float CURVE_FREQUENCY = 0.03f;

    inline float getRoadCenterlineX(float z) {
        return CURVE_AMPLITUDE * sinf(CURVE_FREQUENCY * z);
    }

    inline float getRoadTangentAngle(float z) {
        float derivative = CURVE_AMPLITUDE * CURVE_FREQUENCY * cosf(CURVE_FREQUENCY * z);
        return atanf(derivative);
    }

    inline Vector2 getRoadTangentVector(float z) {
        float angle = getRoadTangentAngle(z);
        return Vector2Normalize({sinf(angle), cosf(angle)});
    }

    inline Vector2 getRoadNormalVector(float z) {
        Vector2 tangent = getRoadTangentVector(z);
        return {-tangent.y, tangent.x};
    }

} 
}
