using namespace std;

#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<sstream>
#include<fstream>
#include<string>
#include<vector>
#include<iomanip>
#include <string.h>
#include "tools.h"

int id,type,Ntot;
double Lminx,Lmaxx,Lminy,Lmaxy,Lminz,Lmaxz;
int ix,iy,iz;
double x,y,z;
double Lx,Ly,Lz;

long int k=0;
long int Kmax;
int Kstart;
const int Nmaxx=10;
const int Npolymax=2000;

int Nmax,Nbeads;
atom *atoms; 
int DeltaT=10000;
double Wr=0.0;

string dummy; 
//
//
double Polymer[Nmaxx][Npolymax][3];
double Polymer0[Nmaxx][Npolymax][3];
double COM[Nmaxx][3];
double COM0[Nmaxx][3];
int lp;
// double allunsWr[Nmaxx][Npolymax];
double localWr[Nmaxx][Npolymax];

// double ComputeSegmUnSignedWrithe(int n1, int n2, int n3);
double ComputeLocalUnSignedWrithe(int n1, int n2, int n3);

int main(int argc, char* argv[]){

cout << "Write argv[1]: datain -- \nargv[2] = dataout no extension \targv[3]=max # of frames -- argv[4]=nebads; argv[5]=window size"<< endl;
    
Kmax=atoi(argv[3]);
//
//
Nbeads=atoi(argv[4]);
Nmax=1;
//
//
lp=atof(argv[5]);
Kstart = int(atoi(argv[6]));

//
//OUTFILE
//
stringstream writeFileWr;
writeFileWr <<"Writhe_" << argv[2]<< ".lp"<<lp<<".dat";
ofstream writeWr(writeFileWr.str().c_str());
writeWr << "#Timestep BeadNumber 1DUnsignedWrithe 3DLocalWrithe "<< endl;
//
// stringstream writeFileWrmax;
// writeFileWrmax <<"WritheMAX_" << argv[2]<< ".lp"<<lp<<".dat";
// ofstream writeWrmax(writeFileWrmax.str().c_str());
// writeWrmax << "#Timestep Max1DUnsignedWrithe ArgMax1DUnsignedWrithe Max3DLocalWrithe ArgMax3DLocalWrithe "<< endl;

//
/////////////////////////////////////////////////////
//MAIN LOOP OVER TIME!!!
////////////////////////////////////////////////////
for(k=0;k<Kmax;k++){
ifstream indata;
stringstream readFile;
readFile.str("");
readFile.clear();
long int timet;
long long int sum = int(k+Kstart);
//if(k==0)readFile << argv[1] <<sum;
//had to fix this as previously forgot "sum" and "" around 0000, so was only reading first config repeatedly - joseph
readFile << argv[1] <<sum <<"0000";
indata.open(readFile.str().c_str());
cout << readFile.str().c_str()<<endl;
if(!indata.good()){cout << "AHAHAHAHHA"<<endl; cin.get();cin.get();}

//read 10 lines
for(int i=0;i<10;i++){
if(i==1) {
indata >> timet;
timet = sum*DeltaT/10000.0;
cout << "time " << timet <<endl;
}
if(i==3) {
indata >> Ntot;
cout << "Ntot " << Ntot<<endl;
}
if(i==5) {
indata >> Lminx >> Lmaxx;
cout << "L " << Lminx<< " " << Lmaxx <<endl;
Lx = Lmaxx-Lminx;
cout << "Lx " << Lx <<endl;
}
if(i==6) {
indata >> Lminy >> Lmaxy;
cout << "L " << Lminy << " " << Lmaxy <<endl;
Ly = Lmaxy-Lminy;
cout << "Ly " << Ly <<endl; 
}
if(i==7) {
indata >> Lminz >> Lmaxz;
cout << "L " << Lminz << " " << Lmaxz <<endl;
Lz = Lmaxz-Lminz;
cout << "Lz " << Lz <<endl;
}

else getline(indata,dummy);
cout << dummy<<endl;
}

//////////////////////////
//READ FILES
////////////////////////////
int id,mol,type;
cout << "reading .. "<<endl;
for(int n=0; n<Ntot; n++){

	indata >> id>> type >> x>>y>>z>>ix>>iy>>iz;
    int npoly = floor((double)(id-1)/(double)(Nbeads));
	int nb = (id-1)%Nbeads;

	Polymer[npoly][nb][0]=x+Lx*ix;
	Polymer[npoly][nb][1]=y+Ly*iy;
	Polymer[npoly][nb][2]=z+Lz*iz;
	//COM
	COM[npoly][0]+=Polymer[npoly][nb][0]*1.0/Nbeads;
	COM[npoly][1]+=Polymer[npoly][nb][1]*1.0/Nbeads;
	COM[npoly][2]+=Polymer[npoly][nb][2]*1.0/Nbeads;
	
	if(k==0){
	Polymer0[npoly][nb][0]=Polymer[npoly][nb][0];
	Polymer0[npoly][nb][1]=Polymer[npoly][nb][1];
	Polymer0[npoly][nb][2]=Polymer[npoly][nb][2];
	}
}
//set initial points
if(k==0){
for(int n=0; n<Nmax; n++){
COM0[n][0]=COM[n][0];
COM0[n][1]=COM[n][1];
COM0[n][2]=COM[n][2];
}
}
cout << "done "<<endl;

//////////////
// Compute Writhe
/////////////////
cout << "Computing Writhe " <<endl;
//cin.get();
//for lp=Nbeads/2 one has the full writhe
// double maxw=0;
// int argmax=0;
// double localmaxw=0;
// int localargmax=0;
for(int n=0;n<Nmax;n++){
cout << "doing polymer .. " << n << endl;
//had to change index because k was used for time loop as well, so changed to kk - joseph
for(int kk=0;kk<Nbeads;kk+=1){
// allunsWr[n][kk]=ComputeSegmUnSignedWrithe(n,kk,lp);
localWr[n][kk]=ComputeLocalUnSignedWrithe(n,kk,lp);
// if (allunsWr[n][kk]>maxw) {
//     maxw = allunsWr[n][kk];
//     argmax = kk;
// }
// if (localWr[n][kk]>localmaxw) {
//     localmaxw = localWr[n][kk];
//     localargmax = kk;
// }
// writeWr << k << " " << kk <<" " << allunsWr[n][kk] << " " << localWr[n][kk] << endl;
writeWr << k << " " << kk << " "  << localWr[n][kk] << endl;

}
// writeWrmax << k << " " << maxw << " " << argmax << " " << localmaxw << " " << localargmax << endl;
// writeWrmax << k << " "  << localmaxw << " " << localargmax << endl;

writeWr << "\n\n" ;
}

}//close loop over read 

return 0; 
}

// double ComputeSegmUnSignedWrithe(int r, int k, int lp){
//     atoms = new atom[Nbeads];
//     for (int i=0;i<Nbeads;i++) {
//         atoms[i].x=Polymer[r][i][0];
//         atoms[i].y=Polymer[r][i][1];
//         atoms[i].z=Polymer[r][i][2];
//     }
//     // calculate writhe for segment k-lp < i < k+lp
//     double Wr=0.0;
//     xyz one,two, three, four,
//     r12,r34,r23,r13,r14,r24,
//     n1,n2,n3,n4,cvec;
//     double omega, n1n2, n2n3, n3n4, n4n1;
//     int i1=k-lp;int i2=k;
    
//     for (int i=i1;i<i2;i++) {  //loop over segments within k-lp < i < k
//         int ib=i;
//         if(i<0)ib=i+Nbeads;
//         if(i>=Nbeads)ib=i%Nbeads;
        
//         int j1=k; int j2=k+lp;
//         for (int j=j1;j<j2;j++) {  //loop over segments k < j < k+lp
//             int jb=j;
//             if(j<0)jb=j+Nbeads;
//             if(j>=Nbeads)jb=j%Nbeads;
            
//             if (ib==jb||jb==ib-1||jb==ib+1||(ib==0&&jb==Nbeads-1)||(ib==Nbeads-1&&jb==0)) {
//                 omega=0.0;
//             } else {
//                 //
//                 if (jb==0) {
//                     three=atoms[Nbeads-1];//D[L-1];
//                 } else{
//                     three=atoms[jb-1];
//                 }
//                 if(ib==0){
//                     one=atoms[Nbeads-1];
//                 }else{
//                     one=atoms[ib-1];
//                 }
                
//                 two=atoms[ib]; four=atoms[jb];
//                 r12=two-one;
//                 r34=four-three;
//                 r23=three-two;
//                 r13=three-one;
//                 r14=four-one;
//                 r24=four-two;
                
//                 n1=r13.cross(r14); //if (n1.length()<SMALL) {cout<<"error in writhe 1 "<<n1.length()<<" ("<<i<<")"<<endl;}
//                 n1.make_unit();
                
//                 n2=r14.cross(r24); //if (n2.length()<SMALL) {cout<<"error in writhe 2 "<<n1.length()<<" ("<<i<<")"<<endl;}
//                 n2.make_unit();
                
//                 n3=r24.cross(r23); //if (n3.length()<SMALL) {cout<<"error in writhe 3 "<<n1.length()<<" ("<<i<<")"<<endl;}
//                 n3.make_unit();
                
//                 n4=r23.cross(r13); //if (n4.length()<SMALL) {cout<<"error in writhe 4 "<<n1.length()<<" ("<<i<<")"<<endl;}
//                 n4.make_unit();
                
//                 n1n2=n1.dot(n2); fix_roundoff11(n1n2);
//                 n2n3=n2.dot(n3); fix_roundoff11(n2n3);
//                 n3n4=n3.dot(n4); fix_roundoff11(n3n4);
//                 n4n1=n4.dot(n1); fix_roundoff11(n4n1);
                
//                 cvec=r34.cross(r12);
                
//                 omega = ( asin( n1n2 ) + asin( n2n3 ) + asin( n3n4 ) + asin( n4n1 ) )*1;
                
//             }
//             Wr+= omega/(4.0*PI);
//         } //loop over j
//     } //loop over beads i
//     Wr*=2.0;
//     delete [] atoms;
//     return Wr;
// }

double ComputeLocalUnSignedWrithe(int r, int k, int lp){
    atoms = new atom[Nbeads];
    for (int i=0;i<Nbeads;i++) {
        atoms[i].x=Polymer[r][i][0];
        atoms[i].y=Polymer[r][i][1];
        atoms[i].z=Polymer[r][i][2];
    }
    // calculate writhe for segment k-lp/2 < i < k+lp/2 over entire contour - local method of non-local entanglement
    double Wr=0.0;
    xyz one,two, three, four,
    r12,r34,r23,r13,r14,r24,
    n1,n2,n3,n4,cvec;
    double omega, n1n2, n2n3, n3n4, n4n1;
    int i1=k-(lp/2);int i2=k+(lp/2);
    
    for (int i=i1;i<i2;i++) {  //loop over segments within k-lp/2 < i < k+lp/2
        int ib=i;
        if(i<0)ib=i+Nbeads;
        if(i>=Nbeads)ib=i%Nbeads;
        
        int j1=0; int j2=Nbeads;
        for (int j=j1;j<j2;j++) {  //loop over entire contour 0 < j < Nbeads
            int jb=j;
            if(j<0)jb=j+Nbeads;
            if(j>=Nbeads)jb=j%Nbeads;
            
            if (ib==jb||jb==ib-1||jb==ib+1||(ib==0&&jb==Nbeads-1)||(ib==Nbeads-1&&jb==0)) {
                omega=0.0;
            } else {
                //
                if (jb==0) {
                    three=atoms[Nbeads-1];//D[L-1];
                } else{
                    three=atoms[jb-1];
                }
                if(ib==0){
                    one=atoms[Nbeads-1];
                }else{
                    one=atoms[ib-1];
                }
                
                two=atoms[ib]; four=atoms[jb];
                r12=two-one;
                r34=four-three;
                r23=three-two;
                r13=three-one;
                r14=four-one;
                r24=four-two;
                
                n1=r13.cross(r14); //if (n1.length()<SMALL) {cout<<"error in writhe 1 "<<n1.length()<<" ("<<i<<")"<<endl;}
                n1.make_unit();
                
                n2=r14.cross(r24); //if (n2.length()<SMALL) {cout<<"error in writhe 2 "<<n1.length()<<" ("<<i<<")"<<endl;}
                n2.make_unit();
                
                n3=r24.cross(r23); //if (n3.length()<SMALL) {cout<<"error in writhe 3 "<<n1.length()<<" ("<<i<<")"<<endl;}
                n3.make_unit();
                
                n4=r23.cross(r13); //if (n4.length()<SMALL) {cout<<"error in writhe 4 "<<n1.length()<<" ("<<i<<")"<<endl;}
                n4.make_unit();
                
                n1n2=n1.dot(n2); fix_roundoff11(n1n2);
                n2n3=n2.dot(n3); fix_roundoff11(n2n3);
                n3n4=n3.dot(n4); fix_roundoff11(n3n4);
                n4n1=n4.dot(n1); fix_roundoff11(n4n1);
                
                cvec=r34.cross(r12);
                
                omega = ( asin( n1n2 ) + asin( n2n3 ) + asin( n3n4 ) + asin( n4n1 ) )*1;
                
            }
            Wr+= omega/(4.0*PI);
        } //loop over j
    } //loop over beads i
    Wr*=2.0;
    delete [] atoms;
    return Wr;
}
