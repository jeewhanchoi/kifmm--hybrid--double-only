#ifndef  _VEC3T_H_
#define  _VEC3T_H_
#include <iostream>
#include <assert.h>
#include "reals.h"

using namespace std;
using std::istream;
using std::ostream;
using std::min;
using std::max;
//using std::abs;

//! Common 3D VECtor Template
template <class F>
//! Common 3D Vector template-based class
class Vec3T {
private:
  /*! 3D vector Data of type F */
  F _v[3];
public:
  enum{ X=0, Y=1, Z=2 };
  //------------CONSTRUCTOR AND DESTRUCTOR
  /*! Constructor */
  Vec3T()              { _v[0]=F(0);    _v[1]=F(0);    _v[2]=F(0); }
  /*! Constructor.  Set _v[0], _v[1], _v[2] = f of type F */
  Vec3T(F f)           { _v[0]=f;       _v[1]=f;       _v[2]=f;}
  /*! Constructor.  Set _v[0], _v[1], _v[2] = f of type F */
  Vec3T(const F* f)    { _v[0]=f[0];    _v[1]=f[1];    _v[2]=f[2]; }
  /*! Constructor.  Set _v[0] = a, _v[1] = b, _v[2] = c of type F */
  Vec3T(F a,F b,F c)   { _v[0]=a;       _v[1]=b;       _v[2]=c; }
  /*! Constructor.  Set _v[0] = c._v[0], _v[1] = c._v[1], _v[2] = c._v[2] */
  Vec3T(const Vec3T& c){ _v[0]=c._v[0]; _v[1]=c._v[1]; _v[2]=c._v[2]; }
  /*! Destroy vector */
  ~Vec3T() {}
 //------------POINTER and ACCESS
  /*! Return pointer to _v[0] */
  operator F*()             { return &_v[0]; }
  /*! Return pointer to _v[0] */
  operator const F*() const { return &_v[0]; }
  /*! Return pointer to _v[0] */
  F* array()                { return &_v[0]; }  //access array
  /*! Return ith (0, 1, or 2) element of _v */
  F& operator()(int i)             { assert(i<3); return _v[i]; }
  /*! Return ith (0, 1, or 2) element of _v */
  const F& operator()(int i) const { assert(i<3); return _v[i]; }
  /*! Return ith (0, 1, or 2) element of _v */
  F& operator[](int i)             { assert(i<3); return _v[i]; }
  /*! Return ith (0, 1, or 2) element of _v */
  const F& operator[](int i) const { assert(i<3); return _v[i]; }
  /*! Return x-coordinate (_v[0]) */
  F& x()             { return _v[0];}
  /*! Return y-coordinate (_v[1]) */
  F& y()             { return _v[1];}
  /*! Return z-coordinate (_v[2]) */
  F& z()             { return _v[2];}
  /*! Return x-coordinate (_v[0]) */
  const F& x() const { return _v[0];}
  /*! Return y-coordinate (_v[0]) */
  const F& y() const { return _v[1];}
  /*! Return z-coordinate (_v[0]) */
  const F& z() const { return _v[2];}
  //------------ASSIGN
 /*! Overloaded = operator.  this._v[0] = c._v[0], etc. */
  Vec3T& operator= ( const Vec3T& c ) { _v[0] =c._v[0]; _v[1] =c._v[1]; _v[2] =c._v[2]; return *this; }
  /*! Overloaded += operator.  this._v[0] += c._v[0], etc. */
  Vec3T& operator+=( const Vec3T& c ) { _v[0]+=c._v[0]; _v[1]+=c._v[1]; _v[2]+=c._v[2]; return *this; }
  /*! Overloaded -= operator.  this._v[0] -= c._v[0], etc. */
  Vec3T& operator-=( const Vec3T& c ) { _v[0]-=c._v[0]; _v[1]-=c._v[1]; _v[2]-=c._v[2]; return *this; }
  /*! Overloaded *= operator.  this._v[0] *= c._v[0], etc. */
  Vec3T& operator*=( const F& s )     { _v[0]*=s;       _v[1]*=s;       _v[2]*=s;       return *this; }
  /*! Overloaded /= operator.  this._v[0] /= c._v[0], etc. */
  Vec3T& operator/=( const F& s )     { _v[0]/=s;       _v[1]/=s;       _v[2]/=s;       return *this; }
  //-----------LENGTH...
  /*! L-1 norm:  sabsolute value of sum of elements in vector */
  F l1( void )     const  { F sum=F(0); for(int i=0; i<3; i++) sum=sum+abs(_v[i]); return sum; }
  /*! L-infinity norm:  max of elements in vector */
  F linfty( void ) const  { F cur=F(0); for(int i=0; i<3; i++) cur=max(cur,abs(_v[i])); return cur; }
  /*! L-2 norm:  square root of sum of elements */
  F l2( void )     const  { F sum=F(0); for(int i=0; i<3; i++) sum=sum+_v[i]*_v[i]; return sqrt(sum); }
  /*! Length = L-2 norm */
  F length( void ) const  { return l2(); }
  /*! Unit vector in director of this */
  Vec3T dir( void )    const  { F a=l2(); return (*this)/a; }
};
//-----------BOOLEAN OPS
/*! Boolean == overloaded operator returns true if a==b */
template <class F> inline bool operator==(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(int i=0; i<3; i++)   res = res && (a(i)==b(i));  return res;
}
/*! Boolean != overloaded operator returns true if a!=b */
template <class F> inline bool operator!=(const Vec3T<F>& a, const Vec3T<F>& b) {
  return !(a==b);
}
/*! Boolean > overloaded operator returns true if a > b in BOTH x and y-directions */
template <class F> inline bool operator> (const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(int i=0; i<3; i++)   res = res && (a(i)> b(i));  return res; 
}
/*! Boolean < overloaded operator returns true if a < b in BOTH x and y-directions */
template <class F> inline bool operator< (const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(int i=0; i<3; i++)   res = res && (a(i)< b(i));  return res; 
}
/*! Boolean >= overloaded operator returns true if a >= b in BOTH x and y-directions */
template <class F> inline bool operator>=(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(int i=0; i<3; i++)	res = res && (a(i)>=b(i));  return res; 
}
/*! Boolean <= overloaded operator returns true if a <= b in BOTH x and y-directions */
template <class F> inline bool operator<=(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(int i=0; i<3; i++)   res = res && (a(i)<=b(i));  return res; 
}

//-----------NUMERICAL OPS
/*! Overloaded "-" numerical operation.  Return negation of vector a */
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = -a[i]; return r;
}
/*! Overloaded "+" numerical operation.  Return addition of components of a and b */
template <class F> inline Vec3T<F> operator+ (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = a[i]+b[i]; return r; 
}
/*! Overloaded "-" numerical operation.  Return subtraction of compoenets of b from a */
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = a[i]-b[i]; return r;
}
/*! Overloaded "*" numerical operation.  Return scaling of components of a by scl */
template <class F> inline Vec3T<F> operator* (F scl, const Vec3T<F>& a) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
/*! Overloaded "*" numerical operation.  Return scaling of components of a by scl */
template <class F> inline Vec3T<F> operator* (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
/*! Overloaded "/" numerical operation.  Return scaling of components of a by 1/scl */
template <class F> inline Vec3T<F> operator/ (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = a[i]/scl;  return r;
}
/*! Overloaded "*" numerical operation.  Return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] */
template <class F> inline F operator* (const Vec3T<F>& a, const Vec3T<F>& b) {
  F sum=F(0); for(int i=0; i<3; i++) sum=sum+a(i)*b(i); return sum;
}
/*! Dot-product.  Return a*b = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] */
template <class F> inline F dot       (const Vec3T<F>& a, const Vec3T<F>& b) {
  return a*b;
}
/*! Overloaded cross product operator.  Returns a x b = vector (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]) */
template <class F> inline Vec3T<F> operator^ (const Vec3T<F>& a, const Vec3T<F>& b) {
  return Vec3T<F>(a(1)*b(2)-a(2)*b(1), a(2)*b(0)-a(0)*b(2), a(0)*b(1)-a(1)*b(0)); 
}
/*! Return cross-product */
template <class F> inline Vec3T<F> cross     (const Vec3T<F>& a, const Vec3T<F>& b) { 
  return a^b; 
}
//-------------ew OPS
/*! Return r where r[0] = min(a[0],b[0]), etc. */
template <class F> inline Vec3T<F> min(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = min(a[i], b[i]); return r;
}
/*! Return r where r[0] = max(a[0],b[0]), etc. */
template <class F> inline Vec3T<F> max(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = max(a[i], b[i]); return r;
}
/*! Return r where r[0] = abs(a[0]), etc. */
template <class F> inline Vec3T<F> abs(const Vec3T<F>& a) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = abs(a[i]); return r;
}
/*! Return r where r[0] = a[0]*b[0], etc. */
template <class F> inline Vec3T<F> ewmul(const Vec3T<F>&a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = a[i]*b[i]; return r;
}
/*! Return r where r[0] = a[0]/b[0], etc. */
template <class F> inline Vec3T<F> ewdiv(const Vec3T<F>&a, const Vec3T<F>& b) { 
  Vec3T<F> r;  for(int i=0; i<3; i++) r[i] = a[i]/b[i]; return r;
}
//---------------INOUT
/*! input >> operator.  is >> a[i] */
template <class F> istream& operator>>(istream& is, Vec3T<F>& a) {
  for(int i=0; i<3; i++) is>>a[i]; return is;
}
/*! output << operator.  os << a[i] */
template <class F> ostream& operator<<(ostream& os, const Vec3T<F>& a) { 
  for(int i=0; i<3; i++) os<<a[i]<<" "; return os;
}

//---------------------------------------------------------
/// MOST COMMONLY USED
/*! A 3-D point with the x, y, and z-coordinates represented as doubles */
typedef Vec3T<real_t> Point3;
/*! A 3-D index with the three indices represented as integers */ 
typedef Vec3T<int>    Index3;

#endif
