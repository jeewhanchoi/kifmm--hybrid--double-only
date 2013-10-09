#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <set>
#include <queue>
#include <math.h>
#include <vector>

#include "util.h"
#include "input.h"
#include "vec3t.h"
#include "node.h"
#include "reals_aligned.h"

using namespace std;

static
  void
usage__ (const char* use)
{
  fprintf (stderr, "usage: %s <N> <distribution> <max points per box>\n", use);
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
create (int N, char* distribution, AllNodes* All_N) 
{

  int coincide = getenv__coincide();

  if (0==strcmp(distribution,"uniform")) {
    fprintf (stderr, "Using uniform distribution for sources and targets\n");
    for (int i = 0; i < N; i++) {
      All_N->sx_orig[i] = (2.0 * drand48() - 1.0);
      All_N->sy_orig[i] = (2.0 * drand48() - 1.0);
      All_N->sz_orig[i] = (2.0 * drand48() - 1.0);
    }
    if (coincide == 0) {
      for (int i = 0; i < N; i++) {
        All_N->tx_orig[i] = (2.0 * drand48() - 1.0);
        All_N->ty_orig[i] = (2.0 * drand48() - 1.0);
        All_N->tz_orig[i] = (2.0 * drand48() - 1.0);
      }
    }
    else {
        All_N->tx_orig = All_N->sx_orig;
        All_N->ty_orig = All_N->sy_orig;
        All_N->tz_orig = All_N->sz_orig;
    /*  for (int i = 0; i < N; i++) {
        All_N->tx_orig[i] = All_N->sx_orig[i];
        All_N->ty_orig[i] = All_N->sy_orig[i];
        All_N->tz_orig[i] = All_N->sz_orig[i];
      }
    */
    }
  } 
  else if (0==strcmp(distribution,"ellipseUniformAngles")) {
    fprintf (stderr, "Using distribution on 1:1:4 ellipse, uniform in angles\n");
    const double r = 0.49;
    const double center [3] = { 0.5, 0.5, 0.5};
    for (int i = 0; i < N; i++) {
      double phi = 2*M_PI*drand48();
      double theta = M_PI*drand48();
      All_N->sx_orig[i] = center[0] + 0.25 * r * sin(theta) * cos(phi);
      All_N->sy_orig[i] = center[1] + 0.25 * r * sin(theta) * sin(phi);
      All_N->sz_orig[i] = center[2] + r * cos(theta);
    }
    for (int i = 0; i < N; i++) {
      double phi = 2*M_PI*drand48();
      double theta = M_PI*drand48();
      All_N->tx_orig[i] = center[0] + 0.25 * r * sin(theta) * cos(phi);
      All_N->ty_orig[i] = center[1] + 0.25 * r * sin(theta) * sin(phi);
      All_N->tz_orig[i] = center[2] + r * cos(theta);
    }
  }
  else
    fprintf (stderr, "Invalid distribution of sources selected");

  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 */
inline
  real_t 
radius()  
{ 
  int root_level = 0;
  return pow (2.0, root_level); 
}

/* ----------------------------------------------------------------------------------------------------------
 */

/* Power function:  2^l.  Uses shifts */
inline 
  int 
pow2 (int l) 
{ 
  assert (l >= 0); 
  return (1 << l); 
}

/* ----------------------------------------------------------------------------------------------------------
 */

/** Radius of a node/box */
//inline
  real_t 
radius (int nodeId, vector<NodeTree>& nodeVec) 
{
  return radius() / real_t (pow2(nodeVec[nodeId].depth));
}

/* ----------------------------------------------------------------------------------------------------------
 */

/** Center of a node/box */
  Point3 
center (int nodeId, vector<NodeTree>& nodeVec) 
{
  Point3 center(0.0);
  int dim = 3;

  Point3 ll (center - Point3 (radius()));
  int tmp = pow2 (nodeVec[nodeId].depth);
  Index3 pathLcl (nodeVec[nodeId].path2Node);
  Point3 res;
  for (int d = 0; d < dim; d++) {
    res(d) = ll(d) + (2 * radius()) * (pathLcl(d) + 0.5) / real_t(tmp);
  }
  return res;
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
child (int nodeId, const Index3& idx, vector<NodeTree>& nodeVec)
{
  assert (idx >= Index3(0) && idx < Index3(2));
  return nodeVec[nodeId].child + (idx(0) * 4 + idx(1) * 2 + idx(2));
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
findgnt(int wntdepth, const Index3& wntpath, vector<NodeTree>& nodeVec)
{
  int cur = 0;  
  Index3 leftpath (wntpath);
  while (nodeVec[cur].depth < wntdepth && nodeVec[cur].child != -1) {
    int dif = wntdepth - nodeVec[cur].depth;
    int tmp = pow2 (dif - 1);
    Index3 choice (leftpath / tmp);
    leftpath -= choice * tmp;
    cur = child (cur, choice, nodeVec);	 
  }  
  return cur;
}

/* ----------------------------------------------------------------------------------------------------------
 */

  bool 
adjacent (int a, int b, vector<NodeTree>& nodeVec)
{
  int md = max (nodeVec[a].depth, nodeVec[b].depth);
  Index3 one(1);
  Index3 acenter ((2 * nodeVec[a].path2Node + one) * pow2 (md - nodeVec[a].depth));
  Index3 bcenter ((2 * nodeVec[b].path2Node + one) * pow2(md - nodeVec[b].depth));
  int aradius = pow2 (md - nodeVec[a].depth);
  int bradius = pow2 (md - nodeVec[b].depth);
  Index3 dif (abs(acenter - bcenter));
  int radius  = aradius + bradius;

  return 
    (dif <= Index3 (radius)) && 
    (dif.linfty() == radius); 
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
dwnOrder (vector<int>& orderBoxesVec, vector<NodeTree>& nodeVec)
{
  orderBoxesVec.clear();
  for (int i = 0; i < nodeVec.size(); i++)
    orderBoxesVec.push_back (i);
  assert (orderBoxesVec.size() == nodeVec.size());
  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
src_tree (int N, int pts_max, AllNodes* All_N)
{
  int level = 0;
  int arr_beg = 0;
  int arr_end = 1;
  int arr_count = 0;
  int dim = 3;
  vector< vector<int> > vecId;  
  vector<int> srcNum; 
  vector<NodeTree>& nodeVec = *All_N->N;
  vector<int>& nodeLevel = *All_N->nodeLevel;

  /* Push root node */ 
  nodeVec.push_back (NodeTree (-1, -1, Index3(0,0,0), 0));
  vecId.push_back (vector<int>());
  vector<int>& curVecId = vecId[0];

  for (int k = 0; k < N; k++) {
    // TODO: Add boundary checking
    curVecId.push_back (k);
  }
  srcNum.push_back (curVecId.size());
  
  nodeLevel.push_back (arr_beg);
  while (arr_beg < arr_end) {
    arr_count = arr_end;
    for (int k = arr_beg; k < arr_end; k++) {
      /* Check if "max" points per box condition is satisfied */
      if (srcNum[k] > pts_max) {
        nodeVec[k].child = arr_count;
        arr_count = arr_count + pow2 (dim);
        /* Divide the parent node/box into 8 child nodes/boxes */
        for (int a = 0; a < 2; a++) {
          for (int b = 0; b < 2; b++) {
            for (int c = 0; c < 2; c++) {
              nodeVec.push_back (NodeTree (k, -1, 2 * nodeVec[k].path2Node + Index3(a,b,c), nodeVec[k].depth + 1) );
              vecId.push_back (vector<int>());
              srcNum.push_back (0);
            }
          }
        }
        /** Get the center of the current node/box */
        Point3 centerCurNode (center(k, nodeVec));
        /** Determine which child each source point in the parent box belongs to */
        for (vector<int>::iterator vecIdIt = vecId[k].begin(); vecIdIt != vecId[k].end(); vecIdIt++) {
          Index3 idx;
          idx(0) = (All_N->sx_orig[(*vecIdIt)] >= centerCurNode(0));
          idx(1) = (All_N->sy_orig[(*vecIdIt)] >= centerCurNode(1));
          idx(2) = (All_N->sz_orig[(*vecIdIt)] >= centerCurNode(2));

          int childNodeId = child (k, idx, nodeVec);
          vecId[childNodeId].push_back (*vecIdIt);
        }
        vecId[k].clear(); 
        /* Get the total number of source points in each node/box */
        for (int a = 0; a < 2; a++) {
          for (int b = 0; b < 2; b++) {
            for (int c = 0; c < 2; c++) {
              int childNodeId = child (k, Index3(a,b,c), nodeVec);
              srcNum[childNodeId] = vecId[childNodeId].size();
            }
          }
        }
      }
    }
    level++;
    arr_beg = arr_end;
    arr_end = arr_count;
    nodeLevel.push_back (arr_beg);
  }

  /** Ordering of the nodes/boxes, in top-down fashion */
  vector<int> orderBoxesVec;  
  dwnOrder (orderBoxesVec, nodeVec);

  /* Tag source nodes/boxes */
  int cnt = 0;
  int sum = 0;
  for (int i = 0; i < orderBoxesVec.size(); i++) {
    int nodeIdx = orderBoxesVec[i];
    if (srcNum[nodeIdx] > 0) {
      nodeVec[nodeIdx].tag = nodeVec[nodeIdx].tag | LET_SRCNODE;
      nodeVec[nodeIdx].srcNodeIdx = cnt;
      cnt++;
      if(nodeVec[nodeIdx].child==-1) {
        nodeVec[nodeIdx].srcBeg = sum;
        nodeVec[nodeIdx].srcNum = srcNum[nodeIdx];
        sum += srcNum[nodeIdx];
        nodeVec[nodeIdx].srcOwnVecIdxs = vecId[nodeIdx];
      }
    }
  }

  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
compute_lists (int nodeId, vector<NodeTree>& nodeVec)
{
  set<int> Uset, Vset, Wset, Xset;
  int curNodeId = nodeId;

  if (nodeVec[curNodeId].parent != -1) {
    int parentNodeId = nodeVec[curNodeId].parent;	 
    Index3 minIdx(0);
    Index3 maxIdx (pow2 (nodeVec[curNodeId].depth));

    for (int i = -2; i < 4; i++) 
      for (int j = -2; j < 4; j++) 
        for (int k = -2; k < 4; k++) {
          Index3 tryPath (2 * nodeVec[parentNodeId].path2Node + Index3(i,j,k) );
          if (tryPath >= minIdx && tryPath <  maxIdx && tryPath != nodeVec[curNodeId].path2Node) {	
            int resNodeId = findgnt (nodeVec[curNodeId].depth, tryPath, nodeVec);
            bool adj = adjacent (resNodeId, curNodeId, nodeVec);
            if (nodeVec[resNodeId].depth < nodeVec[curNodeId].depth) { 
              if (adj) {
                if (nodeVec[curNodeId].child == -1) 
                  Uset.insert(resNodeId);
                else { ; }
              }
              else {
                Xset.insert(resNodeId);
              }
            }
            if (nodeVec[resNodeId].depth == nodeVec[curNodeId].depth) {
              if (!adj) {
                Index3 bb (nodeVec[resNodeId].path2Node - nodeVec[curNodeId].path2Node);
                assert (bb.linfty() <= 3);
                Vset.insert (resNodeId);
              }
              else {
                if (nodeVec[curNodeId].child == -1) {
                  queue<int> rest;
                  rest.push (resNodeId);
                  while (rest.empty() == false) {
                    int frontNodeId = rest.front(); 
                    rest.pop();		
                    if (adjacent (frontNodeId, curNodeId, nodeVec) == false) 
                      Wset.insert (frontNodeId);       
                    else {
                      if (nodeVec[frontNodeId].child == -1) 
                        Uset.insert (frontNodeId);
                      else { 
                        for (int a = 0; a < 2; a++) 
                          for (int b = 0; b < 2; b++) 
                            for (int c = 0; c < 2; c++) 
                              rest.push (child(frontNodeId, Index3(a,b,c), nodeVec));
                      }
                    }
                  }
                }
              }
            }
          }
        }
  }

  if (nodeVec[curNodeId].child == -1) 
    Uset.insert (curNodeId);

  for (set<int>::iterator si = Uset.begin(); si != Uset.end(); si++)
    if (nodeVec[*si].tag & LET_SRCNODE)		
      nodeVec[nodeId].Unodes.push_back (*si);
  for (set<int>::iterator si = Vset.begin(); si != Vset.end(); si++)
    if (nodeVec[*si].tag & LET_SRCNODE)		
      nodeVec[nodeId].Vnodes.push_back (*si);
  for (set<int>::iterator si = Wset.begin(); si != Wset.end(); si++)
    if (nodeVec[*si].tag & LET_SRCNODE)		
      nodeVec[nodeId].Wnodes.push_back (*si);
  for (set<int>::iterator si = Xset.begin(); si != Xset.end(); si++)
    if (nodeVec[*si].tag & LET_SRCNODE)		
      nodeVec[nodeId].Xnodes.push_back (*si);

  return (0);
}

/* ----------------------------------------------------------------------------------------------------------
 */

  int 
trg_tree (int N, AllNodes* All_N)
{
  vector<NodeTree>& nodeVec = *All_N->N;
  vector< vector<int> > vecId; 
  vecId.resize (nodeVec.size());
  vector<int> trgNum;           
  trgNum.resize (nodeVec.size(), 0);

  vector<int>& curVecId = vecId[0];

  for (int k = 0; k < N; k++) { 
    curVecId.push_back (k);
  }
  trgNum[0] = curVecId.size();

  vector<int> orderBoxesVec;  
  dwnOrder (orderBoxesVec, nodeVec);
  for (int i = 0; i < orderBoxesVec.size(); i++) {
    int nodeId = orderBoxesVec[i];
    NodeTree& curNode = nodeVec[nodeId];
    vector<int>& curVecId = vecId[nodeId];	 
    if (curNode.child != -1) { 
      Point3 curCenter (center(nodeId, nodeVec));
      /* Determine which child each target point belong to */    
      for (vector<int>::iterator curVecIdIt = curVecId.begin(); curVecIdIt != curVecId.end(); curVecIdIt++) {
        Index3 idx;
        idx(0) = (All_N->tx_orig[(*curVecIdIt)] >= curCenter(0));
        idx(1) = (All_N->ty_orig[(*curVecIdIt)] >= curCenter(1));
        idx(2) = (All_N->tz_orig[(*curVecIdIt)] >= curCenter(2));
        int childNodeId = child (nodeId, idx, nodeVec);
        vector<int>& childVecId = vecId[childNodeId];
        childVecId.push_back (*curVecIdIt);
      }
      curVecId.clear(); 

      for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
          for (int c = 0; c < 2; c++) {
            int childNodeId = child (nodeId, Index3(a,b,c), nodeVec);
            trgNum[childNodeId] = vecId[childNodeId].size();
          }
        }
      }
    }
  }

  /* Get the total number of target points in each node/box */
  int cnt = 0;
  int sum = 0;
  for (int i = 0; i < orderBoxesVec.size(); i++) {
    int nodeId = orderBoxesVec	[i];
    if (trgNum[nodeId] > 0) { 
      nodeVec[nodeId].tag = nodeVec[nodeId].tag | LET_TRGNODE;
      nodeVec[nodeId].trgNodeIdx = cnt;
      cnt ++;
      if(nodeVec[nodeId].child==-1) { //terminal
        nodeVec[nodeId].trgBeg = sum;
        nodeVec[nodeId].trgNum = trgNum[nodeId];
        sum += trgNum[nodeId];
        nodeVec[nodeId].trgOwnVecIdxs = vecId[nodeId];
      }
    }
  }

  /** Compute U-, V-, W- and X-lists */
  for (int i=0; i < orderBoxesVec.size(); i++) {
    int nodeId = orderBoxesVec[i];
    if (nodeVec[nodeId].tag & LET_TRGNODE) { 
      compute_lists (nodeId, nodeVec);
    }
  }

  return 0;
}

/* ----------------------------------------------------------------------------------------------------------
 * eof
 */

