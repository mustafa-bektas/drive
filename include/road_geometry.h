#pragma once

#include <cmath>
#include "raymath.h" // For sinf, cosf, atanf, PI

namespace CarGame {
namespace RoadGeometry {

    // --- Road Curve Parameters ---
    // Adjust these values to change the shape of the curve
    constexpr float CURVE_AMPLITUDE = 15.0f; // Max horizontal deviation from center (meters)
    constexpr float CURVE_FREQUENCY = 0.03f; // Controls how quickly the road curves along Z

    /**
     * @brief Calculates the X coordinate of the road centerline at a given Z coordinate.
     * @param z The Z coordinate along the road.
     * @return The X coordinate of the centerline at that Z.
     */
    inline float getRoadCenterlineX(float z) {
        return CURVE_AMPLITUDE * sinf(CURVE_FREQUENCY * z);
    }

    /**
     * @brief Calculates the tangent angle (direction) of the road at a given Z coordinate.
     * The angle is measured in radians relative to the positive Z-axis.
     * @param z The Z coordinate along the road.
     * @return The tangent angle in radians.
     */
    inline float getRoadTangentAngle(float z) {
        // The derivative of the centerline function gives the slope (dx/dz)
        float derivative = CURVE_AMPLITUDE * CURVE_FREQUENCY * cosf(CURVE_FREQUENCY * z);
        // The angle is the arctangent of the slope
        return atanf(derivative);
    }

    /**
     * @brief Calculates the normalized tangent vector of the road at a given Z coordinate.
     * @param z The Z coordinate along the road.
     * @return A Vector2 representing the normalized tangent direction (dx, dz).
     */
    inline Vector2 getRoadTangentVector(float z) {
        float angle = getRoadTangentAngle(z);
        // Convert angle to a direction vector (cos(angle) for Z, sin(angle) for X)
        // Note: Raylib typically uses Y-up, but our road is on the XZ plane.
        // The angle is relative to the Z-axis.
        return Vector2Normalize({sinf(angle), cosf(angle)}); // {X component, Z component}
    }

    /**
     * @brief Calculates the normalized normal vector (perpendicular to the tangent) of the road at a given Z coordinate.
     * @param z The Z coordinate along the road.
     * @return A Vector2 representing the normalized normal direction.
     */
    inline Vector2 getRoadNormalVector(float z) {
        Vector2 tangent = getRoadTangentVector(z);
        // Rotate tangent by 90 degrees to get the normal (swap components, negate one)
        return {-tangent.y, tangent.x}; // Perpendicular vector pointing "left" relative to tangent
    }

} // namespace RoadGeometry
} // namespace CarGame
