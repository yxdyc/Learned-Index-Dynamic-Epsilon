//
//

#ifndef LEARNED_INDEX_INDEXMECHANISM_HPP
#define LEARNED_INDEX_INDEXMECHANISM_HPP


// using size_t = unsigned long;


#include <cmath>
#include <limits>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <float.h>




/**
 * A struct that stores a segment.
 * @tparam KeyType the type of the elements that the segment indexes
 * @tparam Floating the floating-point type of the segment's parameters
 */
template<typename KeyType, typename Floating>
struct FittingSegmentModified {
    //static_assert(std::is_floating_point<Floating>());
    KeyType seg_start;              ///< The first key that the segment indexes.
    Floating seg_slope;     ///< The slope of the segment.
    KeyType seg_intercept;              ///< The intercept of the segment index, used for predict pos.
    KeyType seg_end;              ///< The last key that the segment indexes.
    KeyType seg_last_y;              ///< The intercept of the segment index, used for predict pos.


    FittingSegmentModified() = default;

    /**
     * Constructs a new segment.
     * @param slope the slope of the segment
     * @param KeyType the start key of the segment
     * @param intercept the intercept of the lineal segment
     * @param seg_last_y, the position of the last key
     */
    FittingSegmentModified(KeyType start, Floating slope, KeyType intercept, KeyType end, KeyType seg_last_y):
            seg_start(start), seg_slope(slope), seg_intercept(intercept), seg_end(end), seg_last_y(seg_last_y){};

    friend inline bool operator<(const FittingSegmentModified &s, const KeyType k) {
        return s.seg_start < k;
    }

    friend inline bool operator<(const FittingSegmentModified &s1, const FittingSegmentModified &s2) {
        return s1.seg_start < s2.seg_start;
    }

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline Floating operator()(KeyType k) const {
        Floating pos = seg_slope * float(k - seg_start) + float(seg_intercept);
        return pos > Floating(0) ? pos : 0.0;
    }
};





template<typename T>
using LargeSigned = typename std::conditional<std::is_floating_point<T>::value,
        long double,
        typename std::conditional<(sizeof(T) < 8),
                int64_t,
                __int128>::type>::type;

template<typename X, typename Y, typename Floating = double>
class PGMMechanism {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Point {
        SX x{};
        SY y{};

        Point() = default;

        Point(SX x, SY y) : x(x), y(y) {};

        Point operator-(Point p) const {
            return Point{SX(x - p.x), SY(y - p.y)};
        }

        bool operator<(Point p) const {
            return y * p.x < x * p.y;
        }

        bool operator>(Point p) const {
            return y * p.x > x * p.y;
        }

        bool operator==(Point p) const {
            return y * p.x == x * p.y;
        }
    };

    const SY error_fwd;
    const SY error_bwd;
    std::vector<Point> lower;
    std::vector<Point> upper;
    size_t lower_start = 0;
    size_t upper_start = 0;
    size_t points_in_hull = 0;
    Point rectangle[4];

    template<typename P>
    SX cross(const P &O, const P &A, const P &B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    }

public:
    explicit PGMMechanism(size_t error_fwd, size_t error_bwd)
            : error_fwd(error_fwd), error_bwd(error_bwd) {
        upper.reserve(1u << 16);
        lower.reserve(1u << 16);
        rectangle[2].x = -INT32_MAX;
        rectangle[3].x = -INT32_MAX;
    }

    bool add_point(X x, Y y) {
        if (x < rectangle[2].x || x < rectangle[3].x)
            throw std::logic_error("Points must be increasing by x.");

        SX xx = x;
        SY yy = y;

        if (points_in_hull == 0) {
            rectangle[0] = {xx, yy + error_fwd};
            rectangle[1] = {xx, yy - error_bwd};
            ++points_in_hull;
            return true;
        }

        if (points_in_hull == 1) {
            rectangle[2] = {xx, yy - error_bwd};
            rectangle[3] = {xx, yy + error_fwd};
            upper.clear();
            upper.push_back(rectangle[0]);
            upper.push_back(rectangle[3]);
            lower.clear();
            lower.push_back(rectangle[1]);
            lower.push_back(rectangle[2]);
            upper_start = lower_start = 0;
            ++points_in_hull;
            return true;
        }

        Point p1(xx, yy + error_fwd);
        Point p2(xx, yy - error_bwd);
        auto slope1 = rectangle[2] - rectangle[0];
        auto slope2 = rectangle[3] - rectangle[1];
        bool outside_line1 = p1 - rectangle[2] < slope1;
        bool outside_line2 = p2 - rectangle[3] > slope2;

        if (outside_line1 || outside_line2) {
            points_in_hull = 0;
            return false;
        }

        if (p1 - rectangle[1] < slope2) {
            // Find extreme slope
            Point min = lower[lower_start] - p1;
            size_t min_i = lower_start;
            for (size_t i = lower_start + 1; i < lower.size(); i++) {
                auto val = (lower[i] - p1);
                if (val > min)
                    break;
                else {
                    min = val;
                    min_i = i;
                }
            }

            rectangle[1] = lower[min_i];
            rectangle[3] = p1;
            lower_start = min_i;

            // Hull update
            size_t end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end);
            upper.resize(end);
            upper.push_back(p1);
        }

        if (p2 - rectangle[0] > slope1) {
            // Find extreme slope
            Point max = upper[upper_start] - p2;
            size_t max_i = upper_start;
            for (size_t i = upper_start + 1; i < upper.size(); i++) {
                auto val = (upper[i] - p2);
                if (val < max)
                    break;
                else {
                    max = val;
                    max_i = i;
                }
            }

            rectangle[0] = upper[max_i];
            rectangle[2] = p2;
            upper_start = max_i;

            // Hull update
            size_t end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end);
            lower.resize(end);
            lower.push_back(p2);
        }

        ++points_in_hull;
        return true;
    }

    std::pair<double, double> get_intersection() const {
        auto &p0 = rectangle[0];
        auto &p1 = rectangle[1];
        auto &p2 = rectangle[2];
        auto &p3 = rectangle[3];
        auto slope1 = p2 - p0;
        auto slope2 = p3 - p1;

        if (points_in_hull == 1 || slope1 == slope2)
            return std::make_pair(double(p0.x), double(p0.y));

        double a = slope1.x * slope2.y - slope1.y * slope2.x;
        double b = ((p1.x - p0.x) * (p3.y - p1.y) - (p1.y - p0.y) * (p3.x - p1.x)) / a;
        auto i_x = p0.x + b * slope1.x;
        auto i_y = p0.y + b * slope1.y;
        return std::make_pair(i_x, i_y);
    }

    double get_intercept(X key) const {
        std::pair<double, double> intersection = get_intersection();
        //auto[i_x, i_y] = get_intersection();
        std::pair<double, double> slope_range = get_slope_range();
        //auto[min_slope, max_slope] = get_slope_range();
        //auto slope = 0.5 * (min_slope + max_slope);
        auto slope = 0.5 * (slope_range.first + slope_range.second);
        //return i_y - (i_x - key) * slope;
        return intersection.second - (intersection.first - key) * slope;
    }

    std::pair<double, double> get_slope_range() const {
        if (points_in_hull == 1)
            return {0, 1};
        auto min_slope = double(rectangle[2].y - rectangle[0].y) / (rectangle[2].x - rectangle[0].x);
        auto max_slope = double(rectangle[3].y - rectangle[1].y) / (rectangle[3].x - rectangle[1].x);
        return {min_slope, max_slope};
    }

    void reset() {
        points_in_hull = 0;
    }
};
#endif //LEARNED_INDEX_INDEXMECHANISM_HPP
