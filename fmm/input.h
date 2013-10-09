#ifndef __INPUT_H__
#define __INPUT_H__
#include <vector>

#include "vec3t.h"

using namespace std;

/* ----------------------------------------------------------------------------------------------------------
*/

enum {
  LET_SRCNODE = 1,
  LET_TRGNODE = 2
};

struct NodeTree {
    int parent, child, depth, tag;
	  Index3 path2Node;
	 
	 /*! source node index */
	 int srcNodeIdx;
	 /*! source exact beginning index */
	 int srcBeg;
	 /*! source exact number */
	 int srcNum;
	 /*! source own vector of indices */
	 vector<int> srcOwnVecIdxs;

	 /*! target node index */
	 int trgNodeIdx;
	 /*! target exact beginning index */
	 int trgBeg;
	 /*! target exact number */
	 int trgNum;
	 /*! target own vector of indices */
	 vector<int> trgOwnVecIdxs;

    /* Nodes in the U-List */
	  vector<int> Unodes;
	  /* Nodes in the V-List */
	  vector<int> Vnodes;
	  /* Nodes in the W-List */
	  vector<int> Wnodes;
	  /* Nodes in the X-List */
	  vector<int> Xnodes; 
  
    NodeTree (int p, int c, Index3 t, int d):
		  parent(p), child(c), path2Node(t), depth(d), tag(false), srcNodeIdx(0), srcBeg(0), srcNum(0), trgNodeIdx(0),
trgBeg(0), trgNum(0) {;}
};

/* ----------------------------------------------------------------------------------------------------------
*/

int child (int nodeId, const Index3& idx, vector<NodeTree>& nodeVec);

Point3  center (int nodeId, vector<NodeTree>& nodeVec);

real_t radius (int nodeId, vector<NodeTree>& nodeVec);

#endif
