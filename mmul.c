#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#define STEP 64
#define bi 64
#define bj 64 
#define bk 64 


void mmul(float *A, float *B, float *C, int n)
{
// Please modify this function

int i, j, k, it, jt, kt, ii, jj, p, zikuai, z_pnum, z_pid;
int line, zuo, you;
float *buffer, *Bb, *fasong, *huancun, *ans, r;


MPI_Comm_size(MPI_COMM_WORLD, &z_pnum);
MPI_Comm_rank(MPI_COMM_WORLD, &z_pid);

line=n/z_pnum;
buffer = (float*)malloc(sizeof(float)*line*n);
ans = (float*)malloc(sizeof(float)*line*n);
Bb = (float*)malloc(sizeof(float)*n*line);
fasong=(float*)malloc(sizeof(float)*n*n);
huancun=(float*)malloc(sizeof(float)*n*line);

MPI_Scatter(A,line*n,MPI_FLOAT,buffer,line*n,MPI_FLOAT,0,MPI_COMM_WORLD);

if (z_pid==0)
    {
        
        
        for (i=0;i<z_pnum;i++)
        {
        for (k=0; k<n; k++)
        for (j=0; j<line; j++)
        fasong[(i*line)*n+k*line+j]=B[k*n+i*line+j];

        
        }
      MPI_Scatter(fasong,n*line,MPI_FLOAT,Bb,n*line,MPI_FLOAT,0,MPI_COMM_WORLD);   
      
      
      for (p=0; p<z_pnum; p++){
      zikuai=(p+z_pid)%z_pnum;
      
      if(p>0){
      MPI_Send(Bb,n*line,MPI_FLOAT,z_pnum-1,3,MPI_COMM_WORLD);
      MPI_Recv(Bb,n*line,MPI_FLOAT,1,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
       __m128 p0=_mm_set1_ps(0);
			for (it=0; it<line; it+=STEP) {
      for (jt=0; jt<line; jt+=STEP) {
      float res[STEP][STEP] __attribute__ ((aligned (16)));
       for (i=0; i<STEP; i++)
       for (j=0; j<STEP; j+=4)
      _mm_store_ps(&res[i][j],p0);
      
      
      for (kt=0; kt<n; kt+=STEP){
      float Ab[STEP][STEP] __attribute__ ((aligned (16)));
       for (i=it; i<it+STEP; i++) {
			 for (k=kt; k<kt+STEP; k+=4) {
      //Ab[(i-it)*STEP+k-kt]= A[i*n+k];
      __m128 xa = _mm_load_ps(&A[i*n+k]);
      _mm_store_ps(&Ab[i-it][k-kt], xa);
      }
      }
      float bbb[STEP][STEP] __attribute__ ((aligned (16)));
       for (k=kt; k<kt+STEP; k++) {
			 for (j=jt; j<jt+STEP; j+=4) {
      //bbb[(k-kt)*STEP+j-jt]= Bb[k*line+j];
      __m128 xb = _mm_load_ps(&Bb[k*line+j]);
      _mm_store_ps(&bbb[k-kt][j-jt], xb);
      }
      }
      for (i=it; i<it+STEP; i++) {
			for (k=kt; k<kt+STEP; k++) {
      //float temp=Ab[(i-it)*STEP+k-kt];
       //__m128 temp = _mm_load_ps(&Ab[i-it][k-kt]);
       r=Ab[i-it][k-kt];
       __m128 temp = _mm_set1_ps(r);
				for (j=jt; j<jt+STEP; j+=4) {
					//res[(i-it)*STEP+j-jt]+= temp * bbb[(k-kt)*STEP+j-jt];
          __m128 cv = _mm_load_ps(&res[i-it][j-jt]);
          __m128 dv = _mm_load_ps(&bbb[k-kt][j-jt]);                          
          _mm_store_ps(&res[i-it][j-jt], _mm_add_ps(cv, _mm_mul_ps(temp,dv )));                        
					}
				}
			}
      }
      for (i=0; i<STEP; i++)
      for (j=0; j<STEP; j+=4){
      //ans[(it+i)*n+zikuai*line+jt+j]=res[i*STEP+j];
       __m128 ev = _mm_load_ps(&res[i][j]);
       _mm_store_ps(&ans[(it+i)*n+zikuai*line+jt+j], ev);
       }
      }
      }
      }
     
      MPI_Gather(ans,line*n,MPI_FLOAT,C,line*n,MPI_FLOAT,0,MPI_COMM_WORLD);
     
    }

   
    else
    {
      zuo=(z_pnum+z_pid-1)%z_pnum;
      you=(z_pid+1)%z_pnum;
      
      //MPI_Recv(Bb,n*line,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
      //MPI_Recv(buffer,line*n,MPI_FLOAT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
      MPI_Scatter(fasong,n*line,MPI_FLOAT,Bb,n*line,MPI_FLOAT,0,MPI_COMM_WORLD);
      for (p=0; p<z_pnum; p++){
      zikuai=(p+z_pid)%z_pnum;
      
      if(p>0){
      if(z_pid%2==0){
      MPI_Send(Bb,n*line,MPI_FLOAT,zuo,3,MPI_COMM_WORLD);
      MPI_Recv(Bb,n*line,MPI_FLOAT,you,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      else{
      for (i=0; i<n; i++)
      for (j=0; j<line; j++)
      huancun[i*line+j]=Bb[i*line+j];   
      MPI_Recv(Bb,n*line,MPI_FLOAT,you,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Send(huancun,n*line,MPI_FLOAT,zuo,3,MPI_COMM_WORLD);
      }
      }
      
      
      
			__m128 p0=_mm_set1_ps(0); 
			for (it=0; it<line; it+=STEP) {
      for (jt=0; jt<line; jt+=STEP) {
      float res[STEP][STEP] __attribute__ ((aligned (16)));
      for (i=0; i<STEP; i++)
       for (j=0; j<STEP; j+=4)
      _mm_store_ps(&res[i][j],p0);
      
      
      
      for (kt=0; kt<n; kt+=STEP){
      float Ab[STEP][STEP] __attribute__ ((aligned (16)));
       for (i=it; i<it+STEP; i++) {
			 for (k=kt; k<kt+STEP; k+=4) {
      //Ab[(i-it)*STEP+k-kt]= buffer[i*n+k];
      __m128 xa = _mm_load_ps(&buffer[i*n+k]);
      _mm_store_ps(&Ab[i-it][k-kt], xa);
      }
      }
     float bbb[STEP][STEP] __attribute__ ((aligned (16)));
       for (k=kt; k<kt+STEP; k++) {
			 for (j=jt; j<jt+STEP; j+=4) {
      //bbb[(k-kt)*STEP+j-jt]= Bb[k*line+j];
      __m128 xb = _mm_load_ps(&Bb[k*line+j]);
      _mm_store_ps(&bbb[k-kt][j-jt], xb);
      }
      }
      for (i=it; i<it+STEP; i++) {
			for (k=kt; k<kt+STEP; k++) {
      //float temp=Ab[(i-it)*STEP+k-kt];
      //__m128 temp = _mm_load_ps(&Ab[i-it][k-kt]);
       r=Ab[i-it][k-kt];
       __m128 temp = _mm_set1_ps(r);
				for (j=jt; j<jt+STEP; j+=4) {
					//res[(i-it)*STEP+j-jt]+= temp * bbb[(k-kt)*STEP+j-jt];
           __m128 cv = _mm_load_ps(&res[i-it][j-jt]);
          __m128 dv = _mm_load_ps(&bbb[k-kt][j-jt]);                          
          _mm_store_ps(&res[i-it][j-jt], _mm_add_ps(cv, _mm_mul_ps(temp,dv )));     
					}
				}
			}
      }
      for (i=0; i<STEP; i++)
      for (j=0; j<STEP; j+=4){
      //ans[(it+i)*n+zikuai*line+jt+j]=res[i*STEP+j];
       __m128 ev = _mm_load_ps(&res[i][j]);
       _mm_store_ps(&ans[(it+i)*n+zikuai*line+jt+j], ev);
       }
      }
      }
			
      //MPI_Send(ans,line*line, MPI_FLOAT,0,10+zikuai,MPI_COMM_WORLD);
       
    }
    MPI_Gather(ans,line*n,MPI_FLOAT,C,line*n,MPI_FLOAT,0,MPI_COMM_WORLD);
    }




}

