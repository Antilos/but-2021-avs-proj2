/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    unsigned triangleCount = 0;

    Vec3_t<float> zero = (0.0, 0.0, 0.0);
    #pragma omp parallel shared(triangleCount, field) firstprivate(zero) default(none)
    #pragma omp single nowait
    triangleCount =  octree(zero, field, mGridSize);

    // printf("DEBUG: total triangle count:%d\n", trianglesCount);
    return triangleCount;
}

unsigned TreeMeshBuilder::octree(const Vec3_t<float> &position, const ParametricScalarField &field, const unsigned currentGridSize)
{
    // printf("DEBUG: ==========\n");
    // printf("DEBUG: Current Grid Size=%d\n", currentGridSize);
    unsigned totalTriangleCount = 0;

    /*Compute center of the current cube.
    * pos * resolution + (currentGridSize*resolution)/2
    * Multiplying cubes by resolution gives length
    */
    float edgeLength = float(currentGridSize)*mGridResolution;
    float halfEdgeLength = edgeLength/2.0;
    Vec3_t<float> center(
        position.x * mGridResolution + halfEdgeLength,
        position.y * mGridResolution + halfEdgeLength,
        position.z * mGridResolution + halfEdgeLength
    );
    bool empty = evaluateFieldAt(center, field) > mIsoLevel + (sqrt(3.0)/2.0) * edgeLength;

    // printf("DEBUG: empty=%d\n", empty);

    if(empty){
        // printf("DEBUG: Cube is empty\n");
        return 0;
    }else if(currentGridSize <= GRID_SIZE_CUT_OFF){
        // printf("DEBUG: building cube\n");
        // return buildCube(position, field);
        unsigned cubesToBuild = currentGridSize * currentGridSize * currentGridSize;
        unsigned numOfTrianglesBuilt = 0;
        float realGridSize = float(mGridResolution);
        for(size_t i = 0; i < cubesToBuild; ++i){
            // #pragma omp task firstprivate(i) shared(position, field, numOfTrianglesBuilt, realGridSize, mGridResolution) default(none)
            {
            unsigned localX = i % currentGridSize;
            unsigned localY = (i / currentGridSize) % currentGridSize;
            unsigned localZ = i / (currentGridSize * currentGridSize);
            Vec3_t<float> cubeOffset(
                position.x + localX,
                position.y + localY,
                position.z + localZ
            );
            
            unsigned numOfTriangles = buildCube(cubeOffset, field);

            // #pragma omp atomic
            numOfTrianglesBuilt += numOfTriangles;
            }
        }
        // #pragma omp taskwait

        return numOfTrianglesBuilt;
    }else{
        // printf("DEBUG: Splitting tree\n");

        unsigned newGridSize = currentGridSize/2;
        float realGridSize = float(newGridSize);

        //for each corner (easier than computing from just (0,0,0))
        for (Vec3_t<float> corner : sc_vertexNormPos){
            #pragma omp task firstprivate(corner) shared(position, realGridSize, newGridSize, field, totalTriangleCount) default(none)
            {
            Vec3_t<float> newPosition(
                position.x + corner.x * realGridSize,
                position.y + corner.y * realGridSize,
                position.z + corner.z * realGridSize
            );
            unsigned triangleCount = octree(newPosition, field, newGridSize);

            #pragma omp atomic
            totalTriangleCount += triangleCount;
            }
        }
    }

    #pragma omp taskwait
    // printf("DEBUG: returning triangle count = %d\n", totalTriangleCount);
    return totalTriangleCount;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        // printf("DEBUG: NumThreads=%d\n", omp_get_num_threads());
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical(emit_triangle)
    mTriangles.push_back(triangle);
}
