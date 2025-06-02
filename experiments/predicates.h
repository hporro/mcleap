#ifndef PREDICATES_H
#define PREDICATES_H

#ifdef __cplusplus
extern "C"
{
#endif
    void exactinit();
    double orient2d(double *pa, double *pb, double *pc);
    double orient3d(double *pa, double *pb, double *pc, double *pd);
    double incircle(double *pa, double *pb, double *pc, double *pd);
    double insphere(double *pa, double *pb, double *pc, double *pd, double *pe);

#ifdef __cplusplus
}
#endif

#endif // PREDICATES_H