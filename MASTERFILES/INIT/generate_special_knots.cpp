#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<fstream>
#include<sstream>
#include<vector> 
#include <unistd.h>
#include<ctime>
using namespace std;

int main(int argc, char* argv[]){

cout << "This program generates (q,p) Torus knots" <<endl;
cout << "Type 1. output name; 2. N; 3. p; 4. q;" <<endl;

int pk,qk;


ofstream write;
write.open(argv[1]);
int N=atoi(argv[2]);
//knot parameters ////////
// nx=atoi(argv[3]);
// ny=atoi(argv[4]);
// nz=atoi(argv[5]);
// deltax=atol(argv[6]);
// deltay=atol(argv[7]);
// deltaz=atol(argv[8]);

//////////////////////////

double pos_k1[N][3];

write <<"LAMMPS data file from restart file: timestep = 0, procs = 1"<<endl;
write <<endl;

write<< 1*N << " atoms"<<endl;
write<< 1*N << " bonds"<<endl;
write<< 1*N << " angles"<<endl;
write <<endl;

write << 4 << " atom types" <<endl;
write << 1 << " bond types" <<endl;
write << 1 << " angle types" <<endl;
write <<endl;

write << "-80 80"  << " xlo xhi "<<endl; //120 for 1000 unknot, 80 for SH, lower bead number unknots
write << "-80 80"  << " ylo yhi "<<endl;
write << "-80 80"  << " zlo zhi "<<endl;
write <<endl;

write << "Masses "<<endl;
write <<endl;
write << " 1 1 " <<endl;
write << " 2 1 " <<endl;
write << " 3 1 " <<endl;
write << " 4 1 " <<endl;
write <<endl;

write << "Atoms "<<endl;
write <<endl;

// int n;
// // //TO CREATE 4_1 KNOT
// for(n=0;n<N;n++){

//     double t = 2.0*M_PI*n/(double)(N);
//     double r_in=2; //CHANGES INNER CIRCLE OF TORUS
//     double r = cos(2*t)+r_in;
    
//     double r_out=N*1.0/40; //THIS NUMBER IS EMPIRICAL
//                           //CHANGE IT TO CONTROL OUTER CIRCLE OF TORUS
//                           //HENCE DISTANCE BETWEEN BEADS (AIM TO HAVE ~1)
    
//     pos_k1[n][0]=r_out*(r*cos(3*t));
//     pos_k1[n][1]=r_out*(r*sin(3*t));
//     pos_k1[n][2]=r_out*(-sin(4*t));

//     //NEED TO WRITE:
//     //INDEX MOLECULE TYPE X Y Z IX IY IZ
//     write << n+1 << " 1 1 " << pos_k1[n][0] << " " <<  pos_k1[n][1] << " "  << pos_k1[n][2] << " 0 0 0 " <<endl;
// }

int n;
int nx, ny, nz;
double deltax, deltay, deltaz;
//5_2
// nx = 3;
// ny = 2;
// nz = 7;
// deltax = 0.7;
// deltay = 0.2;
// deltaz = 0.0;

//6_1
// nx = 3;
// ny = 2;
// nz = 5;
// deltax = 1.5;
// deltay = 0.2; 
// deltaz = 0.0;

//8_21
nx = 3;
ny = 4;
nz = 7;
deltax = 0.1;
deltay = 0.7; 
deltaz = 0.0;

//TO CREATE LISSAJOUS KNOTS
for(n=0;n<N;n++){

    double t = 2.0*M_PI*n/(double)(N);
    //double r_in=2; //CHANGES INNER CIRCLE OF TORUS
    //double r = cos(2*t)+r_in;
    
    double r_out=2*N*1.0/40; //THIS NUMBER IS EMPIRICAL
                          //CHANGE IT TO CONTROL OUTER CIRCLE OF TORUS
                          //HENCE DISTANCE BETWEEN BEADS (AIM TO HAVE ~1)
    
    pos_k1[n][0]=r_out*cos(nx*t + deltax);
    pos_k1[n][1]=r_out*cos(ny*t + deltay);
    pos_k1[n][2]=r_out*cos(nz*t + deltaz);

    //NEED TO WRITE:
    //INDEX MOLECULE TYPE X Y Z IX IY IZ
    write << n+1 << " 1 1 " << pos_k1[n][0] << " " <<  pos_k1[n][1] << " "  << pos_k1[n][2] << " 0 0 0 " <<endl;
}


//CREATE BONDS BETWEEN BEADS
//THIS IS A LINEAR POLYMER SO N-1 BONDS
int nbonds=1;
write << "Bonds"<<endl;
write <<endl;
for(int n=1;n<=N;n++){
write << nbonds << " " << 1  << " " << n << " " << n%N+1<<endl;
nbonds++;
}

write <<endl;

//CREATE ANGLES BETWEEN BEADS
//THIS IS A LINEAR POLYMER SO N-2 ANGLES
int nangles=1;
write << "Angles"<<endl;
write <<endl;
for(int n=0;n<N;n++){
if(n<N-2)write << nangles << " " << 1  << " " << n+1<< " " << n+2 << " " << n+3<<endl;
if(n==N-2)write << nangles << " " << 1  << " " << n+1 << " " << n+2 << " " << 1<<endl;
if(n==N-1)write << nangles << " " << 1  << " " << n+1 << " " << 1 << " " << 2<<endl;
nangles++;
}


return 0;
}


