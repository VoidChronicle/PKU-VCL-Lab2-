#include <unordered_map>

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"

namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        Engine::SurfaceMesh curr_mesh = input;
        // We do subdivison iteratively.
        for (std::uint32_t it = 0; it < numIterations; ++it) {
            // During each iteration, we first move curr_mesh into prev_mesh.
            Engine::SurfaceMesh prev_mesh;
            prev_mesh.Swap(curr_mesh);
            // Then we create doubly connected edge list.
            DCEL G(prev_mesh);
            if (! G.IsManifold()) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Non-manifold mesh.");
                return;
            }
            // Note that here curr_mesh has already been empty.
            // We reserve memory first for efficiency.
            curr_mesh.Positions.reserve(prev_mesh.Positions.size() * 3 / 2);
            curr_mesh.Indices.reserve(prev_mesh.Indices.size() * 4);
            // Then we iteratively update currently existing vertices.
            for (std::size_t i = 0; i < prev_mesh.Positions.size(); ++i) {
                // Update the currently existing vetex v from prev_mesh.Positions.
                // Then add the updated vertex into curr_mesh.Positions.
                auto v           = G.Vertex(i);
                auto neighbors   = v->Neighbors();
                // your code here:
                int n = neighbors.size();
                float u = n == 3 ? 3.0f / 16.0f : 3.0f / (8.0f * n);
                glm::vec3 newPos = (1.0f - n * u) * prev_mesh.Positions[i];
                for (auto neighbor : neighbors) {
                    newPos += u * prev_mesh.Positions[neighbor];
                }
                // Then add the updated vertex into curr_mesh.Positions.
                curr_mesh.Positions.push_back(newPos);
                // update? index?
            }
            // We create an array to store indices of the newly generated vertices.
            // Note: newIndices[i][j] is the index of vertex generated on the "opposite edge" of j-th
            //       vertex in the i-th triangle. 
            std::vector<std::array<std::uint32_t, 3U>> newIndices(prev_mesh.Indices.size() / 3, { ~0U, ~0U, ~0U });
            // Iteratively process each halfedge.
            for (auto e : G.Edges()) {
                // newIndices[face index][vertex index] = index of the newly generated vertex
                newIndices[G.IndexOf(e->Face())][e->EdgeLabel()] = curr_mesh.Positions.size();
                auto eTwin                                       = e->TwinEdgeOr(nullptr);
                // eTwin stores the twin halfedge.
                if (! eTwin) {
                    // When there is no twin halfedge (so, e is a boundary edge):
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    // 取中点
                    glm::vec3 newPos = 0.5f * (prev_mesh.Positions[e->From()] + prev_mesh.Positions[e->To()]);
                    curr_mesh.Positions.push_back(newPos);


                } else {
                    // When the twin halfedge exists, we should also record:
                    //     newIndices[face index][vertex index] = index of the newly generated vertex
                    // Because G.Edges() will only traverse once for two halfedges,
                    //     we have to record twice.
                    newIndices[G.IndexOf(eTwin->Face())][e->TwinEdge()->EdgeLabel()] = curr_mesh.Positions.size();
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    glm::vec3 pos1 = prev_mesh.Positions[e->From()];
                    glm::vec3 pos2 = prev_mesh.Positions[e->To()];
                    glm::vec3 pos3 = prev_mesh.Positions[e->OppositeVertex()];
                    glm::vec3 pos4 = prev_mesh.Positions[eTwin->OppositeVertex()];
                    glm::vec3 newPos = 0.375f * (pos1 + pos2) + 0.125f * (pos3 + pos4);
                    curr_mesh.Positions.push_back(newPos);

                }
            }

            // Here we've already build all the vertices.
            // Next, it's time to reconstruct face indices.
            for (std::size_t i = 0; i < prev_mesh.Indices.size(); i += 3U) {
                // For each face F in prev_mesh, we should create 4 sub-faces.
                // v0,v1,v2 are indices of vertices in F.
                // m0,m1,m2 are generated vertices on the edges of F.
                auto v0           = prev_mesh.Indices[i + 0U];
                auto v1           = prev_mesh.Indices[i + 1U];
                auto v2           = prev_mesh.Indices[i + 2U];
                auto [m0, m1, m2] = newIndices[i / 3U];
                // Note: m0 is on the opposite edge (v1-v2) to v0.
                // Please keep the correct indices order (consistent with order v0-v1-v2)
                //     when inserting new face indices.
                // toInsert[i][j] stores the j-th vertex index of the i-th sub-face.
                std::uint32_t toInsert[4][3] = {
                    // your code here:
                    v0,m2,m1,
                    m0,m1,m2,
                    v2,m1,m0,
                    v1,m0,m2
                };
                // Do insertion.
                curr_mesh.Indices.insert(
                    curr_mesh.Indices.end(),
                    reinterpret_cast<std::uint32_t *>(toInsert),
                    reinterpret_cast<std::uint32_t *>(toInsert) + 12U
                );
            }

            if (curr_mesh.Positions.size() == 0) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Empty mesh.");
                output = input;
                return;
            }
        }
        // Update output.
        output.Swap(curr_mesh);
    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // Copy.
        output = input;
        // Reset output.TexCoords.
        output.TexCoords.resize(input.Positions.size(), glm::vec2 { 0 });

        // Build DCEL.
        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::Parameterization(..): non-manifold mesh.");
            return;
        }

        // Set boundary UVs for boundary vertices.
        // your code here: directly edit output.TexCoords
        /*
        输出的 output.TexCoords 应该保存每个顶点的 UV 坐标，坐标的数值需要保证在 [0,1] 之内。
        这里我们提供算法的大概流程供大家参考：
        1. 为初始 input Mesh 建立半边数据结构，检查网格上的边界点
        （具体来说，只被一个三角形面包含的边，其两个端点被称为边界点）；
        2. 初始化边界点上的 UV 坐标，可以选择初始化为正方形边界或者为圆边界；
        3. 迭代求解中间点上的 UV 坐标，简单起见使用平均系数作为仿射组合系数，
        随后通过 Gauss-Seidel 迭代求解方程组。*/
        int numBoundary = 0;
        for (std::size_t i = 0; i < input.Positions.size(); ++i) {
            auto v = G.Vertex(i);
            if (v->OnBoundary()) {             
                numBoundary++;
                //printf("%lld %f %f\n", i, 0.5,0.5);
                //auto f = input.TexCoords;
                ///std::cout << i << " " << input.TexCoords[i].x << std::endl;
                //printf("%lld %f %f\n", i, input.TexCoords[i].x, input.TexCoords[i].y);
                //为什么，竟然不能打印？？？？
                // 好像input没有TexCoords

                //printf("%lld %f %f %f\n", i, input.Positions[i].x, input.Positions[i].y, input.Positions[i].z);
            }
        }
        
        int num = 0; 
        for (std::size_t i = 0; i < input.Positions.size(); ++i) {
            auto v = G.Vertex(i);
            if (v->OnBoundary()) {
                float y = input.Positions[i].y;
                float x = input.Positions[i].x;
                float norm = sqrt(x * x + y * y);
                x          = (x * 0.5 / norm + 0.5);
                y          = (y * 0.5 / norm + 0.5);
                /*x *= 4;
                y *= 4;*/

                //arctan(input.Positions[i].y/input.Positions[i].x)
                /*float angle = atan2(input.Positions[i].y, input.Positions[i].x);
                glm::vec2 pos = { angle / (2 * glm::pi<float>()) + 0.5f, 0.5f };
                output.TexCoords[i] = pos;*/

                glm::vec2 pos       = { x, y };
                output.TexCoords[i] = pos;
                //printf("%f %f\n", pos.x, pos.y);
                num++;
            }
        }  
        //圆形
        
        /*for (std::size_t i = 0; i < input.Positions.size(); ++i) {
            auto v = G.Vertex(i);
            if (v->OnBoundary()) {
                glm::vec2 pos       = { 0.5f + 0.5f * cos(2 * num * glm::pi<float>() / numBoundary),
                                        0.5f + 0.5f * sin(2 * num * glm::pi<float>() / numBoundary) };
                output.TexCoords[i] = pos;
                printf("%f %f\n", pos.x, pos.y);
                num++;
            }
        }*/
        /*for (std::size_t i = 0; i < input.Positions.size(); ++i) {
            auto v = G.Vertex(i);
            if (v->OnBoundary()) {
                glm::vec2 pos       = { 0.5f + 0.5f * cos(2 * i * glm::pi<float>() / input.Positions.size()),
                                        0.5f + 0.5f * sin(2 * i * glm::pi<float>() / input.Positions.size()) };
                output.TexCoords[i] = pos;
                printf("%f %f\n", pos.x, pos.y);
                num++;
            }
        }*/
        //正方形
        /*int   l     = input.Positions.size();
        float delta = 4.0f / l;
        for (std::size_t i = 0; i < l; ++i) {
            auto v = G.Vertex(i);
            if (v->OnBoundary()) {
                if (i > 3 * l / 4) {
                    glm::vec2 pos       = { 0.0f, 1.0f - delta * (i - 3 * l / 4) };
                    output.TexCoords[i] = pos;
                    printf("%f %f\n", pos.x, pos.y);
                } else if (i > l / 2) {
                    glm::vec2 pos       = { 1.0f - delta * (i - l / 2), 1.0f };
                    output.TexCoords[i] = pos;
                    printf("%f %f\n", pos.x, pos.y);
                } else if (i > l / 4) {
                    glm::vec2 pos       = { 1.0f, delta * (i - l / 4) };
                    output.TexCoords[i] = pos;
                    printf("%f %f\n", pos.x, pos.y);
                } else {
                    glm::vec2 pos       = { delta * i, 0.0f };
                    output.TexCoords[i] = pos;
                    printf("%f %f\n", pos.x, pos.y);
                }
            }
        }*/
        /* int l     = numBoundary;
        int   i     = 0;
        float delta = 4.0f / l;
        for (std::size_t j = 0; j < input.Positions.size(); ++j) {
            auto v = G.Vertex(j);
            if (v->OnBoundary()) {
                glm::vec2 pos = { 0, 0 }; 
                if (i > 3 * l / 4) {
                    pos       = { 0.0f, 1.0f - delta * (i - 3 * l / 4) };
                } else if (i > l / 2) {
                    pos       = { 1.0f - delta * (i - l / 2), 1.0f };
                } else if (i > l / 4) {
                    pos       = { 1.0f, delta * (i - l / 4) };
                } else {
                    pos       = { delta * i, 0.0f };
                }
                output.TexCoords[j] = pos;
                //printf("%f %f\n", pos.x, pos.y);
                i++;
            }
        }*/
        
        // Solve equation via Gauss-Seidel Iterative Method.
        for (int k = 0; k < numIterations; ++k) {
            // your code here:
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                auto v = G.Vertex(i);
                if (!v->OnBoundary()) {
                    auto neighbors = v->Neighbors();
                    //v=average(neighbors)
                    glm::vec2 sum = { 0, 0 };
                    int       numNeighbors = 0;
                    for (auto neighbor : neighbors) {
                        numNeighbors++;
                        sum += output.TexCoords[neighbor];
                    }
                    sum /= numNeighbors;
                    output.TexCoords[i] = sum ;
                }
            }
        }
    }

    //bool isupdated[32768] = {};

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float simplification_ratio) {

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-watertight mesh.");
            return;
        }

        // Copy.
        output = input;

        /*printf("%lld num of v\n", G.NumOfVertices());
        printf("%lld num of f\n", G.NumOfFaces());*/

        // Compute Kp matrix of the face f.
        auto UpdateQ {
            [&G, &output] (DCEL::Triangle const * f) -> glm::mat4 {
                glm::mat4 Kp;
                // your code here:
                // if f satisfy : ax+by+cz+d=0,a^2+b^2+c^2=1
                // then Kp=p*p.T,p=[a,b,c,d]
                // else Kp=0
                auto v0 = output.Positions[f->VertexIndex(0)];
                auto v1 = output.Positions[f->VertexIndex(1)];
                auto v2 = output.Positions[f->VertexIndex(2)];
                glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                float     d      = -glm::dot(normal, v0);
                glm::vec4 p        = { normal, d };
                Kp               = glm::outerProduct(p, p);
                return Kp;
            }
        };

        // The struct to record contraction info.
        struct ContractionPair {
            DCEL::HalfEdge const * edge;            // which edge to contract; if $edge == nullptr$, it means this pair is no longer valid
            glm::vec4              targetPosition;  // the targetPosition $v$ for vertex $edge->From()$ to move to
            float                  cost;            // the cost $v.T * Qbar * v$
        };

        // Given an edge (v1->v2), the positions of its two endpoints (p1, p2) and the Q matrix (Q1+Q2),
        //     return the ContractionPair struct.
        static constexpr auto MakePair {
            [] (DCEL::HalfEdge const * edge,
                glm::vec3 const & p1,
                glm::vec3 const & p2,
                glm::mat4 const & Q
            ) -> ContractionPair {
                // your code here:
                // edge =nullptr;
                // targetPosition:v=glm::inverse([[q11,q12,q13,q14],[q21,q22,q23,q24],[q31,q32,q33,q34],[0,0,0,1]])*[0,0,0,1]
                // cost:v.T*Q*v
                if (! edge) {
                    printf("error: task3:edge=nullptr\n");
                    return {};
                }
                glm::mat4 Q1   = Q;
                Q1[0][3]          = 0;
                Q1[1][3]          = 0;
                Q1[2][3]          = 0;
                Q1[3][3]          = 1;
                //先列坑死我了
                /*printf("Log:%s", glm::to_string(mat).c_str());*/ 
                /*for (int i = 0; i <= 3; i++) {x
                    for (int j = 0; j <= 3; j++) {
                        printf("%f ", Q1[i][j]);
                    }
                    printf("\n");
                }
                printf("\n");*/
                //det(Q1)

                
                glm::vec4 v    = glm::vec4(0, 0, 0, 1);
                float det = glm::determinant(Q1);
                if (det<1e-3 && det>-1e-3) {
                    //printf("error: task3:det(Q1)=0\n");
                    glm::vec3 v0 = (p1 + p2);
                    v0 /= 2;
                    v = glm::vec4(v0, 1);
                    
                } else {
                    //printf("ok: task3:det(Q1)!=0\n");
                    glm::mat4 Qbar = glm::inverse(Q1);
                    v    = Qbar * glm::vec4(0, 0, 0, 1);
                }
                
                

                /*glm::vec3 v0             = (p1 + p2);
                v0 /= 2;
                v = glm::vec4(v0, 1);*/
                float cost = glm::dot(v, Q * v);

                return { edge, v, cost };
                //(return edge or nullptr????)
                //printf to check the matrix 
            }
        };

        // pair_map: map EdgeIdx to index of $pairs$
        // pairs:    store ContractionPair
        // Qv:       $Qv[idx]$ is the Q matrix of vertex with index $idx$
        // Kf:       $Kf[idx]$ is the Kp matrix of face with index $idx$
        std::unordered_map<DCEL::EdgeIdx, std::size_t> pair_map; 
        std::vector<ContractionPair>                  pairs; 
        std::vector<glm::mat4>                         Qv(G.NumOfVertices(), glm::mat4(0));
        std::vector<glm::mat4>                         Kf(G.NumOfFaces(),    glm::mat4(0));

        // Initially, we compute Q matrix for each faces and it accumulates at each vertex.
        for (auto f : G.Faces()) {
            auto Q                 = UpdateQ(f);
            Qv[f->VertexIndex(0)] += Q;
            Qv[f->VertexIndex(1)] += Q;
            Qv[f->VertexIndex(2)] += Q;
            Kf[G.IndexOf(f)]       = Q;
        }

        pair_map.reserve(G.NumOfFaces() * 3);
        pairs.reserve(G.NumOfFaces() * 3 / 2);

        // Initially, we make pairs from all the contractable edges.
        for (auto e : G.Edges()) {
            if (! G.IsContractable(e)) continue;
            auto v1                            = e->From();
            auto v2                            = e->To();
            auto pair                          = MakePair(e, input.Positions[v1], input.Positions[v2], Qv[v1] + Qv[v2]);
            pair_map[G.IndexOf(e)]             = pairs.size();
            pair_map[G.IndexOf(e->TwinEdge())] = pairs.size();
            pairs.emplace_back(pair);
        }

        // Loop until the number of vertices is less than $simplification_ratio * initial_size$.
        while (G.NumOfVertices() > simplification_ratio * Qv.size()) {
            /*for (int i = 0; i < 32768; i++) {
                isupdated[i] = false;
            }*/

            // Find the contractable pair with minimal cost.
            std::size_t min_idx = ~0;
            for (std::size_t i = 1; i < pairs.size(); ++i) {
                if (! pairs[i].edge) continue;
                /*if (i <= 100) {
                    printf("%f %f cost \n", pairs[i].cost, pairs[min_idx].cost);
                }*/
                if (!~min_idx || pairs[i].cost < pairs[min_idx].cost) {
                   /* printf("i:%lld\n", i);
                    printf("%d\n", pairs[i].edge->From());*/
                    if (G.IsContractable(pairs[i].edge)) {
                        min_idx = i;
                        
                    } else {
                        pairs[i].edge = nullptr;
                        /*printf("falg3");*/
                    }
                    /*if (i >= 3000) printf("flag0: 3000\n");
                    if (min_idx >= 2876) {
                        printf("flag0: 2876\n");
                        printf("min_idx:%lld\n", min_idx);
                    }*/
                }
            }
            if (!~min_idx) break;

            //if (min_idx >= 2876) printf("flag1: 2876\n");
            
            // top:    the contractable pair with minimal cost
            // v1:     the reserved vertex
            // v2:     the removed vertex
            // result: the contract result
            // ring:   the edge ring of vertex v1
            ContractionPair & top    = pairs[min_idx];
            auto               v1     = top.edge->From();
            auto               v2     = top.edge->To();
            auto               result = G.Contract(top.edge);
            auto               ring   = G.Vertex(v1)->Ring();

            top.edge             = nullptr;            // The contraction has already been done, so the pair is no longer valid. Mark it as invalid.
            output.Positions[v1] = top.targetPosition; // Update the positions.

            // We do something to repair $pair_map$ and $pairs$ because some edges and vertices no longer exist.
            for (int i = 0; i < 2; ++i) {
                DCEL::EdgeIdx removed           = G.IndexOf(result.removed_edges[i].first);
                DCEL::EdgeIdx collapsed         = G.IndexOf(result.collapsed_edges[i].second);
                pairs[pair_map[removed]].edge   = result.collapsed_edges[i].first;
                pairs[pair_map[collapsed]].edge = nullptr;
                pair_map[collapsed]             = pair_map[G.IndexOf(result.collapsed_edges[i].first)];
            }

            // For the two wing vertices, each of them lose one incident face.
            // So, we update the Q matrix.
            Qv[result.removed_faces[0].first] -= Kf[G.IndexOf(result.removed_faces[0].second)];
            Qv[result.removed_faces[1].first] -= Kf[G.IndexOf(result.removed_faces[1].second)];

            // For the vertex v1, Q matrix should be recomputed.
            // And as the position of v1 changed, all the vertices which are on the ring of v1 should update their Q matrix as well.
            
            //bool isupdated[G.NumOfVertices()] = {};
            
            //if (min_idx >= 2876) printf("flag2: 2876\n");

            //isupdated[v1]         = true;
            Qv[v1] = glm::mat4(0);
            for (auto e : ring) {
                // your code here:
                //     1. Compute the new Kp matrix for $e->Face()$.
                //     2. According to the difference between the old Kp (in $Kf$) and the new Kp (computed in step 1),
                //        update Q matrix of each vertex on the ring (update $Qv$).
                //     3. Update Q matrix of vertex v1 as well (update $Qv$).
                //     4. Update $Kf$.
                
                // For the vertex v1, Q matrix should be recomputed.
                // And as the position of v1 changed, all the vertices which are on the ring of v1 should update their Q matrix as well.
                
                
                    // 1. Compute the new Kp matrix for e->Face().
                    auto newKp = UpdateQ(e->Face());
                    // 2. According to the difference between the old Kp (in Kf) and the new Kp (computed in step 1),
                    //    update Q matrix of each vertex on the ring (update Qv).
                    auto oldKp = Kf[G.IndexOf(e->Face())];
                    auto diffKp = newKp - oldKp;
                    auto v = e->From(); 
                    //isupdated[v] = true;
                    Qv[v] += diffKp;
                    v = e->To();
                    Qv[v] += diffKp;
                    
                    // 3. Update Q matrix of vertex v1 as well (update Qv).
                    Qv[v1] += newKp;
                    // 4. Update Kf.
                    Kf[G.IndexOf(e->Face())] = newKp;
                
            }

            // Finally, as the Q matrix changed, we should update the relative $ContractionPair$ in $pairs$.
            // Any pair with the Q matrix of its endpoints changed, should be remade by $MakePair$.
            // your code here:
            //for (auto e : G.Edges()) {
            //    if (! G.IsContractable(e)) continue;
            //    auto v1                            = e->From();
            //    auto v2                            = e->To();
            //    //if v2 has changed Q in the last process
            //    if (isupdated[v2]) {
            //        auto pair                          = MakePair(e, input.Positions[v1], input.Positions[v2], Qv[v1] + Qv[v2]);
            //        pairs[pair_map[G.IndexOf(e)]].targetPosition = pair.targetPosition;
            //        pairs[pair_map[G.IndexOf(e)]].cost           = pair.cost;
            //        
            //    }
            //}
            for (auto e : G.Vertex(v1)->Ring()) {
                auto v1   = e->From();
                auto ring = G.Vertex(v1)->Ring();
                for (auto e1 : ring) {
                    auto e2 = e1->NextEdge();
                    if (! G.IsContractable(e2)) {
                        pairs[pair_map[G.IndexOf(e2)]].edge = nullptr;
                    } else {
                        auto v2                                                  = e1->To();
                        auto pair                                                = MakePair(e2, output.Positions[v1], output.Positions[v2], Qv[v1] + Qv[v2]);
                        pairs[pair_map[G.IndexOf(e2)]].targetPosition = pair.targetPosition;
                        pairs[pair_map[G.IndexOf(e2)]].cost           = pair.cost;
                    }
                }
            }

        }
        //感谢尹锦润的参考与debug

        // In the end, we check if the result mesh is watertight and manifold.
        if (! G.DebugWatertightManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Result is not watertight manifold.");
        }

        auto exported = G.ExportMesh();
        output.Indices.swap(exported.Indices);
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // Define function to compute cotangent value of the angle v1-vAngle-v2
        static constexpr auto GetCotangent {
            [] (glm::vec3 vAngle, glm::vec3 v1, glm::vec3 v2) -> float {
                glm::vec3 e1 = v1 - vAngle;
                glm::vec3 e2 = v2 - vAngle;
                float dotProduct = glm::dot(e1, e2);
                float crossProductLength = glm::length(glm::cross(e1, e2));
                float r =dotProduct / crossProductLength;
                /*if (r < 0) {
                    r = -r;
                }*/
                if (r < 0.5f) {
                    //printf("1\n");
                    r = 0.5f;
                }
                return r;

                //just for fun
                /*glm::vec3 a = v1 - vAngle;
                glm::vec3 b = v2 - vAngle;
                float l1 = sqrt(glm::dot(a, a)), l2 = sqrt(glm::dot(b, b));
                float vcos = glm::dot(a, b) / (l1*l2), vsin = sqrt(1 - vcos * vcos);
                return fabs(vcos / vsin);*/
            }
        };

        //printf("%f", GetCotangent({ 1, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }));

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-watertight mesh.");
            return;
        }

        Engine::SurfaceMesh prev_mesh;
        prev_mesh.Positions = input.Positions;
        for (std::uint32_t iter = 0; iter < numIterations; ++iter) {
            Engine::SurfaceMesh curr_mesh = prev_mesh;
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                // your code here: curr_mesh.Positions[i] = ...
                auto v = G.Vertex(i);
                auto neighbors = v->Neighbors();
                float     sumWeight = 0;
                glm::vec3 newPos    = glm::vec3(0);
                glm::vec3 v0        = curr_mesh.Positions[i];
                for (auto neighbor : neighbors) {
                    glm::vec3 neighborPos = curr_mesh.Positions[neighbor];
                    auto      neighborNeighbors = G.Vertex(neighbor)->Neighbors();
                    int       hasfind           = 0;
                    for (auto neighbor2 : neighborNeighbors) {
                        /*float tmpweight[2] = { 0, 0 };
                        glm::vec3 tmppos[2]    = {
                            {0, 0, 0},
                            {0, 0, 0}
                        };*/
                        for (auto neighbor3 : neighbors) {
                            if (neighbor2 == neighbor3) {
                                glm::vec3 neighbor2Pos = curr_mesh.Positions[neighbor2];
                                float     weight       = useUniformWeight ? 0.5f : GetCotangent(neighbor2Pos, neighborPos, v0);
                                //0.5f因为加两次
                                /*tmpweight[hasfind] = weight;
                                tmppos[hasfind]    = neighborPos;*/
                                sumWeight += weight;
                                newPos += weight * neighborPos;
                                hasfind++;
                                break;
                            }
                        }
                        if (hasfind >= 2) {
                            /*float weight = tmpweight[0] + tmpweight[1];
                            if (weight > 1e-3) {
                                sumWeight += weight;
                                newPos += (tmpweight[0]  + tmpweight[1]) * tmppos[1];
                            }*/
                            break;
                        }
                    }
                    
                }
                if (sumWeight > 1e-9 /* || sumWeight < -1e-9*/) {
                    newPos /= sumWeight;

                } 
                else {
                    newPos = v0;
                    printf("error: task4:cotangent:sumweight=0\n");
                }
                curr_mesh.Positions[i] = newPos * lambda + v0 * (1 - lambda);
            }
            // Move curr_mesh to prev_mesh.
            prev_mesh.Swap(curr_mesh);
        }
        // Move prev_mesh to output.
        output.Swap(prev_mesh);
        // Copy indices from input.
        output.Indices = input.Indices;
    }
    

    //bool pos[102][102][102];
    // bloody hell!this fucking VS do not support variable length array
    // 沟槽的VS不能使用变长数组，xmake也会更改配置不能编译
    // 什么垃圾组合，依托臭不可闻的构式
    //int v[101][101][101]; 
    /******************* 5. Marching Cubes *****************/
    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        /*printf("%d\n", n);*/
        // your code here:
        
        //for (int x = 0; x <= n+1; ++x) {
        //    for (int y = 0; y <= n+1; ++y) {
        //        for (int z = 0; z <= n+1; ++z) {
        //            glm::vec3 p = { x * dx, y * dx, z * dx };
        //            p            = grid_min + p;
        //            pos[x][y][z] = 0;
        //            if (sdf(p) >= 0) {
        //                pos[x][y][z] = 1;
        //            }
        //            //printf("%d ", pos[x][y][z]);
        //        }
        //    }
        //}
        //printf("\n");
        //for (int x = 0; x <= n; ++x) {
        //    for (int y = 0; y <= n; ++y) {
        //        for (int z = 0; z <= n; ++z) {
        //            v[x][y][z] = 0;
        //            for (int i = 0; i < 8; i++) {
        //                //v[x][y][z] = 0;
        //                //这句话：我是傻逼
        //                /*printf("%d %d %d %d %d %d %d \n",x,y,z,i,x + (i & 1),y + ((i >> 1) & 1),z + (i >> 2));
        //                printf("%d\n", pos[x + (i & 1)][y + ((i >> 1) & 1)][z + (i >> 2)]);*/
        //                if (pos[x + (i & 1)][y + ((i >> 1) & 1)][z + (i >> 2)]) {
        //                    v[x][y][z] += (1 << i);
        //                    //printf("%d\n", v[x][y][z]);
        //                }
        //            }
        //            //printf("%d\n", v[x][y][z]);
        //        }
        //    }
        //} 
        
        for (int x = 0; x <= n; ++x) {
            for (int y = 0; y <= n; ++y) {
                for (int z = 0; z <= n; ++z) {
                    int idx = 0;
                    //printf("%d ", v[x][y][z]);
                    
                    glm::vec3 p[8] = {};
                    float     d[8] = {};
                    for (int i = 0; i < 8; i++) {                       
                        p[i] = { (x + (i & 1)) * dx, (y + ((i >> 1) & 1)) * dx, (z + (i >> 2)) * dx };
                        p[i]  = grid_min + p[i];
                        d[i] = sdf(p[i]);
                        if (d[i] >= 0) {
                            idx += (1 << i);
                            
                            // printf("%d\n", v[x][y][z]);
                        }

                    }

                    /*if (idx != v[x][y][z]) {
                        printf("%d %d %d %d %d\n", x,y,z,idx, v[x][y][z]);
                    }*/
                    
                    //I am so proud of this!
                    // after 0.5-grid,watch the class,see 资料s,and finally write by hands and mind
                    //show the previous img that use 0.5
                    float edgeTable[12][3] = {
                        {d[0] / (d[0] - d[1]),   0,   0},
                        {d[2] / (d[2] - d[3]),   1,   0},
                        {d[4] / (d[4] - d[5]),   0,   1},
                        {d[6] / (d[6] - d[7]),   1,   1},
                        {  0, d[0] / (d[0] - d[2]),   0},
                        {  0, d[4] / (d[4] - d[6]),   1},
                        {  1, d[1] / (d[1] - d[3]),   0},
                        {  1, d[5] / (d[5] - d[7]),   1},
                        {  0,   0, d[0] / (d[0] - d[4])},
                        {  1,   0, d[1] / (d[1] - d[5])},
                        {  0,   1, d[2] / (d[2] - d[6])},
                        {  1,   1, d[3] / (d[3] - d[7])}
                    };
                    //???? divided by zero

                    /*float edgeTable[12][3] = {
                        {0.5,   0,   0},
                        {0.5,   1,   0},
                        {0.5,   0,   1},
                        {0.5,   1,   1},
                        {  0, 0.5,   0},
                        {  0, 0.5,   1},
                        {  1, 0.5,   0},
                        {  1, 0.5,   1},
                        {  0,   0, 0.5},
                        {  1,   0, 0.5},
                        {  0,   1, 0.5},
                        {  1,   1, 0.5}
                    };*/                    

                    if (idx == 0 || idx == 255) {
                        continue;
                    }
                    for (int i = 0; i < 16; i += 3) {
                        if (c_EdgeOrdsTable[idx][i] == -1) {
                            break;
                        }
                        glm::vec3 p1 = { x + edgeTable[c_EdgeOrdsTable[idx][i]][0], y + edgeTable[c_EdgeOrdsTable[idx][i]][1], z + edgeTable[c_EdgeOrdsTable[idx][i]][2] };
                        glm::vec3 p2 = { x + edgeTable[c_EdgeOrdsTable[idx][i + 1]][0], y + edgeTable[c_EdgeOrdsTable[idx][i + 1]][1], z + edgeTable[c_EdgeOrdsTable[idx][i + 1]][2] };
                        glm::vec3 p3 = { x + edgeTable[c_EdgeOrdsTable[idx][i + 2]][0], y + edgeTable[c_EdgeOrdsTable[idx][i + 2]][1], z + edgeTable[c_EdgeOrdsTable[idx][i + 2]][2] };
                        output.Positions.push_back(grid_min + dx * p1);
                        output.Positions.push_back(grid_min + dx * p2);
                        output.Positions.push_back(grid_min + dx * p3);
                        // too much points.is it unefficent?
                        output.Indices.push_back(output.Positions.size() - 1);
                        output.Indices.push_back(output.Positions.size() - 2);
                        output.Indices.push_back(output.Positions.size() - 3);
                    }
                }
            }
        }
    }
} // namespace VCX::Labs::GeometryProcessing
