#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// Needed on MsWindows
#define NOMINMAX
#include <windows.h>
#endif // Win32 platform

#include <openGL/gl.h>
#include <openGL/glu.h>
// Download glut from: http://www.opengl.org/resources/libraries/glut/
#include <GLUT/glut.h>

#include "float2.h"
#include "float3.h"
#include "float4.h"
#include "float4x4.h"
#include <vector>
int maxDepth = 7;
float3 goldRI = float3(0.21,0.485,1.29);
float3 goldEC = float3(3.13,2.23,1.76);
float3 silverRI = float3(0.15,0.14,0.13);
float3 silverEC = float3(3.7,3.11,2.47);

// simple material class, with object color, and headlight shading
class Material

{
public:
    
    virtual float3 getColor(
                            float3 position,
                            float3 normal,
                            float3 viewDir)
    {
        return normal;
    }
    virtual float3 shade(
                         float3 normal,
                         float3 viewDir, float3 lightDir, float3 powerDensity, float3 hitPosition)
    {
        return normal;
    }
};




// Skeletal camera class.
class Camera
{
    float3 eye;		//< world space camera position
    float3 lookAt;	//< center of window in world space
    float3 right;	//< vector from window center to window right-mid (in world space)
    float3 up;		//< vector from window center to window top-mid (in world space)
    
public:
    Camera()
    {
        eye = float3(0, 0, 3);
        lookAt = float3(0, 0, 2);
        right = float3(1, 0, 0);
        up = float3(0, 1, 0);
    }
    float3 getEye()
    {
        return eye;
    }
    // compute ray through pixel at normalized device coordinates
    float3 rayDirFromNdc(const float2 ndc) {
        return (lookAt - eye
                + right * ndc.x
                + up    * ndc.y
                ).normalize();
    }
};

// Ray structure.
class Ray
{
public:
    float3 origin;
    float3 dir;
    Ray(float3 o, float3 d)
    {
        origin = o;
        dir = d;
    }
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
    Hit()
    {
        t = -1;
    }
    float t;				//< Ray paramter at intersection. Negative means no valid intersection.
    float3 position;		//< Intersection coordinates.
    float3 normal;			//< Surface normal at intersection.
    Material* material;		//< Material of intersected surface.
};

// Object abstract base class.
class Intersectable
{
protected:
    Material* material;
public:
    Intersectable(Material* material):material(material) {}
    virtual Hit intersect(const Ray& ray)=0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.
class QuadraticRoots
{
public:
    float t1;
    float t2;
    // Solves the quadratic a*a*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and set members t1 and t2 to store the roots.
    QuadraticRoots(float a, float b, float c)
    {
        float discr = b * b - 4.0 * a * c;
        if ( discr < 0 ) // no roots
        {
            t1 = -1;
            t2 = -1;
            return;
        }
        float sqrt_discr = sqrt( discr );
        t1 = (-b + sqrt_discr)/2.0/a;
        t2 = (-b - sqrt_discr)/2.0/a;
    }
    // Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
    float getLesserPositive()
    {
        return ((0 < t1 && t1 < t2)||t2<0)?t1:t2;
    }
};

// Object realization.
class Sphere : public Intersectable
{
    float3 center;
    float radius;
public:
    Sphere(const float3& center, float radius, Material* material):
    Intersectable(material),
    center(center),
    radius(radius)
    {
    }
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        float3 diff = ray.origin - center;
        float a = ray.dir.dot(ray.dir);
        float b = diff.dot(ray.dir) * 2.0;
        float c = diff.dot(diff) - radius * radius;
        return QuadraticRoots(a, b, c);
        
    }
    float3 getNormalAt(float3 r)
    {
        return (r - center).normalize();
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal
        float t = solveQuadratic(ray).getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        
        return hit;
    }
    
};

float3 reflect(  	float3 inDir,
               float3 normal)
{
    float cosa = -normal.dot(inDir);
    float3 perp = -normal * cosa;
    float3 parallel = inDir - perp;
    return parallel - perp;
};

class Plane : public Intersectable
{
    float3 ro;
    float3 n;

public:
    Plane(const float3& point, float3 normal, Material* material):
       Intersectable(material),
    
    ro(point), n(normal)
    {}
    
        Hit intersect(const Ray& ray){
        Hit hit;
        float numerator = (ro-ray.origin).dot(n);
        float denominator = ray.dir.dot(n);
        hit.t=numerator/denominator;
        hit.position = ray.origin + ray.dir * hit.t;
        hit.normal = n;
        hit.material = material;
        return hit;
    
    }
};




class HeadlightMaterial : public Material {
    float3 frontFaceColor;
    float3 backFaceColor;
public:
    HeadlightMaterial(float3 frontfaceColor,
                      float3 backfaceColor  ):
    frontFaceColor(frontFaceColor),
    backFaceColor(backFaceColor){}
    HeadlightMaterial():
    frontFaceColor(float3::random()),
    backFaceColor(float3::random())
    {}
    virtual float3 getColor(
                            float3 position,
                            float3 normal,
                            float3 viewDir) {
        //implement headlight shading formula here
        if(viewDir.dot(normal)<0){
            return backFaceColor*normal.dot(-viewDir);
            
        }
        
        return frontFaceColor*normal.dot(viewDir);
    }
};

class Quadric : public Intersectable
{
 

    float4x4 coeffs;
public:

    Quadric(Material* material, float4x4 coeffs):
    Intersectable(material), coeffs(coeffs)
    {
        
    }
    
    Quadric(Material* material):
    Intersectable(material)
    {
        coeffs = float4x4();
}
    QuadraticRoots solveQuadratic(const Ray& ray)
    {
        float4 dir_h = float4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float4 e_h=float4(ray.origin.x, ray.origin.y, ray.origin.z, 1);
        
            float a = dir_h.dot(coeffs*dir_h);
            float b = dir_h.dot(coeffs*e_h)+(e_h.dot(coeffs*dir_h));
            float c = e_h.dot(coeffs*e_h);
        
        return QuadraticRoots(a,b,c);
    }
    float3 getNormalAt(float3 r)
    {
        float4 hitPosition = float4(r.x, r.y, r.z, 1);
        float4 normal = float4((coeffs*hitPosition) + (hitPosition * coeffs)).normalize();
        float3 normalThree = float3(normal.x,normal.y,normal.z);
        return normalThree;
       
    }
    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal
        float t = solveQuadratic(ray).getLesserPositive();
        
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = getNormalAt(hit.position);
        return hit;
    }
    
    Quadric* transform(float4x4 t){
        coeffs = t.invert()*(coeffs*(t.invert().transpose()));
        return this;
    };
    
    
    bool contains(float3 r)
    {
        float4 rhomo(r);
        // evaluate implicit eq
        // return true if negative
        // return false if positive
        float cont = rhomo.dot(coeffs*rhomo);
        if(cont<0){
            return false;
        }
        return true;
    };
    
    // infinite slab, ideal for clipping
    Quadric* parallelPlanes() {
        coeffs = float4x4::identity();
        coeffs._00 = 0;
        coeffs._11 = 1;
        coeffs._22 = 0;
        coeffs._33 = -1;
        return this;
    }

    
    Quadric* sphere(){
        coeffs._33=-1;
        return this;
    };
    
    Quadric* cylinder(){
        coeffs._00=1;
        coeffs._11=0;
        coeffs._22=1;
        coeffs._33=-1;
        return this;
    };
    
    Quadric* cone(){
        coeffs._00=1;
         coeffs._11=-1;
        coeffs._33=0;
        return this;
    };
    
    Quadric* paraboloid(){
        coeffs._11=0;
        coeffs._13=-1;
        coeffs._33=0;
        return this;
    };
    
    Quadric* hyperboloid(){
        coeffs._00=1;
        coeffs._11=-1;
        coeffs._22=1;
        coeffs._33=-1;
        return this;
    };
    
    Quadric* hyperbolicPara(){
        coeffs._00=1;
        coeffs._13=-1;
        coeffs._22=-1;
        coeffs._33=0;
        return this;
    };
    
    Quadric* hyperbolicCyl(){
        coeffs._00=-1;
        coeffs._11=0;
        coeffs._22=0;
        coeffs._33=1;
        return this;
    };
    
    
};

class  Diffuse : public Material
{
    int wave;
    float3 kd = float3(1,1,1);
public:
    Diffuse(float3 kd, int wave):kd(kd), wave(wave){}
    
    float3 snoiseGrad(float3 r)
    {
        unsigned int x = 0x0625DF73;
        unsigned int y = 0xD1B84B45;
        unsigned int z = 0x152AD8D0;
        float3 f = float3(0, 0, 0);
        for(int i=0; i<32; i++)
        {
            float3 s( x/(float)0xffffffff,
                     y/(float)0xffffffff,
                     z/(float)0xffffffff);
            f += s * cos(s.dot(r));
            x = x << 1 | x >> 31;
            y = y << 1 | y >> 31;
            z = z << 1 | z >> 31;
        }
        return f * (1.0 / 64.0);
    }

    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 hitPosition)
    {
        normal+=snoiseGrad(hitPosition*wave)*3;
        normal = normal.normalize();
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        return kd * lightPowerDensity * cosTheta;
    }
};


class PhongBlinn : public Material {
    float3 ks;
    float shininess;
public:
    PhongBlinn(float3 ks, float shininess):ks(ks),shininess(shininess){}
    float3 shade(float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity,
                 float3 hitPosition)
    {
        float3 halfway =
        (viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        return lightPowerDensity * ks
        * pow(cosDelta, shininess);
    }
};

class Plastic : public Material {
    float3 ks;
        float3 kd = float3(1,1,1);
    float shininess;
public:
    Plastic(float3 ks, float shininess):ks(ks),shininess(shininess){}
    float3 shade(float3 normal, float3 viewDir,
                 float3 lightDir, float3 lightPowerDensity, float3 hitPosition)
    {
        float3 halfway =
        (viewDir + lightDir).normalize();
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) return float3(0,0,0);
        return (kd*lightPowerDensity*cosTheta)+lightPowerDensity * ks
        * pow(cosDelta, shininess);
    }
};

class Metal : public Material {
    float3 r0;
public:
    Metal(float3  refractiveIndex, float3  extinctionCoefficient){
        float3 rim1 = refractiveIndex - float3(1,1,1);
        float3 rip1 = refractiveIndex + float3(1,1,1);
        float3 k2 = extinctionCoefficient * extinctionCoefficient;
        r0 = (rim1*rim1 + k2)/ (rip1*rip1 + k2);
    }
    
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 face)
    {
        return  float3(0,0,0);
    }
    struct Event{
        float3 reflectionDir;
        float3 reflectance;

    };
    Event evaluateEvent(float3 inDir, float3 normal) {
        Event e;
        float cosa = -normal.dot(inDir);
        float3 perp = -normal * cosa;
        float3 parallel = inDir - perp;
        e.reflectionDir = parallel - perp;
        e.reflectance = r0 + (float3(1,1,1)-r0) * pow(1 - cosa, 5);
        return e; }
};

class BumpyMetal : public Material {
    float3 r0;
    int wave;
public:
    BumpyMetal(float3  refractiveIndex, float3  extinctionCoefficient, int wavePassed){
        float3 rim1 = refractiveIndex - float3(1,1,1);
        float3 rip1 = refractiveIndex + float3(1,1,1);
        float3 k2 = extinctionCoefficient * extinctionCoefficient;
        r0 = (rim1*rim1 + k2)/ (rip1*rip1 + k2);
        wave = wavePassed;
    }
    float3 snoiseGrad(float3 r)
    {
        unsigned int x = 0x0625DF73;
        unsigned int y = 0xD1B84B45;
        unsigned int z = 0x152AD8D0;
        float3 f = float3(0, 0, 0);
        for(int i=0; i<32; i++)
        {
            float3 s( x/(float)0xffffffff,
                     y/(float)0xffffffff,
                     z/(float)0xffffffff);
            f += s * cos(s.dot(r));
            x = x << 1 | x >> 31;
            y = y << 1 | y >> 31;
            z = z << 1 | z >> 31;
        }
        return f * (1.0 / 64.0);
    }
    
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 hitPosition)
    {
 
        return  float3(0,0,0);
 
    }
    struct Event{
        float3 reflectionDir;
        float3 reflectance;
        float3 hitPosition;
    };
    Event evaluateEvent(float3 inDir, float3 normal,float3 hitPosition) {
        normal+=snoiseGrad(hitPosition*wave);
        normal = normal.normalize();
        Event e;
        float cosa = -normal.dot(inDir);
        float3 perp = -normal * cosa;
        float3 parallel = inDir - perp;
        e.reflectionDir = parallel - perp;
        e.reflectance = r0 + (float3(1,1,1)-r0) * pow(1 - cosa, 5);
        return e; }
};



class Wood : public Material
{
    
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    Wood()
    {
        scale = 16;
        turbulence = 500;
        period = 8;
        sharpness = 10;
    }
    
    float snoise(float3 r) {
        unsigned int x = 0x0625DF73;
        unsigned int y = 0xD1B84B45;
        unsigned int z = 0x152AD8D0;
        float f = 0;
        for(int i=0; i<32; i++) {
            float3 s(x/(float)0xffffffff,
                     y/(float)0xffffffff,
                     z/(float)0xffffffff);
            f += sin(s.dot(r));
            x = x << 1 | x >> 31;
            y = y << 1 | y >> 31;
            z = z << 1 | z >> 31;
        }
        return f / 64.0 + 0.5;
    }
    virtual float3 getColor(
                            float3 position,
                            float3 normal,
                            float3 viewDir)
    {
        //return normal;
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
        w -= int(w);
        return (float3(1, 0.3, 0) * w + float3(0.35, 0.1, 0.05) * (1-w)) * normal.dot(viewDir);
    }
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 hitPosition)
    {
        float3 kd = getColor(hitPosition, normal, viewDir);
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        return kd * lightPowerDensity * cosTheta;
    }
};

class Marble : public Material
{
    float scale;
    float turbulence;
    float period;
    float sharpness;
public:
    Marble()
    {
        scale = 32;
        turbulence = 50;
        period = 32;
        sharpness = 1;
    }
    
    float snoise(float3 r) {
        unsigned int x = 0x0625DF73;
        unsigned int y = 0xD1B84B45;
        unsigned int z = 0x152AD8D0;
        float f = 0;
        for(int i=0; i<32; i++) {
            float3 s(	x/(float)0xffffffff,
                     y/(float)0xffffffff,
                     z/(float)0xffffffff);
            f += sin(s.dot(r));
            x = x << 1 | x >> 31;
            y = y << 1 | y >> 31;
            z = z << 1 | z >> 31;
        }
        return f / 64.0 + 0.5;
    }
    virtual float3 getColor(
                            float3 position,
                            float3 normal,
                            float3 viewDir)
    {
        //return normal;
        float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence;
        w = pow(sin(w)*0.5+0.5, 4);
        return (float3(0, 0, 1) * w + float3(1, 1, 1) * (1-w)) * normal.dot(viewDir);
    }
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 hitPosition)
    {
        float3 kd = getColor(hitPosition, normal, viewDir);
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) return float3(0,0,0);
        return kd * lightPowerDensity * cosTheta;
    }
};



class Dielectric : public Material {
    float  refractiveIndex;
    float  r0;
public:
    Dielectric(float refractiveIndex): refractiveIndex(refractiveIndex) {
        r0 = (refractiveIndex - 1)*(refractiveIndex - 1)
        / (refractiveIndex + 1)*(refractiveIndex + 1);  }
    
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 hitPosition)
    {
        
        return float3(0,0,0);
    }
    struct Event{
        float3 reflectionDir;
        float3 refractionDir;
        float reflectance;
        float transmittance;
    };
    Event evaluateEvent(float3 inDir, float3 normal) {
        Event e;
            float cosa = -normal.dot(inDir);
            float3 perp = -normal * cosa;
            float3 parallel = inDir - perp;
            e.reflectionDir = parallel - perp;
            
            float ri = refractiveIndex;
            if (cosa < 0) { cosa = -cosa; normal = -normal; ri = 1/ri; }
            float disc = 1 - (1 - cosa * cosa) / ri / ri;
            if(disc < 0)
                e.reflectance = 1;
            else {
                float cosb = sqrt(disc);
                e.refractionDir = parallel / ri - normal * cosb;
                e.reflectance = r0 + (1 - r0) * pow(1 - cosa, 5);
            }
            e.transmittance = 1 - e.reflectance;
        return e;  }
};


class Gemstone : public Material {
    float  refractiveIndex;
    float  r0;
public:
    Gemstone(float refractiveIndex): refractiveIndex(refractiveIndex) {
        r0 = (refractiveIndex - 1)*(refractiveIndex - 1)
        / (refractiveIndex + 1)*(refractiveIndex + 1);  }
    
    float3 shade(
                 float3 normal,
                 float3 viewDir,
                 float3 lightDir,
                 float3 lightPowerDensity, float3 hitPosition)
    {
        
        return float3(.1,0,.01);
    }
    struct Event{
        float3 reflectionDir;
        float3 refractionDir;
        float reflectance;
        float transmittance;
    };
    Event evaluateEvent(float3 inDir, float3 normal, float distance) {
        
        Event e;
        float cosa = -normal.dot(inDir);
        float3 perp = -normal * cosa;
        float3 parallel = inDir - perp;
        e.reflectionDir = parallel - perp;
        
        float ri = refractiveIndex;
        if (cosa < 0) { cosa = -cosa; normal = -normal; ri = 1/ri; }
        float disc = 1 - (1 - cosa * cosa) / ri / ri;
        if(disc < 0)
            e.reflectance = 1;
        else {
            float cosb = sqrt(disc);
            e.refractionDir = parallel / ri - normal * cosb;
            e.reflectance = (r0 + (1 - r0) * pow(1 - cosa, 5));
        }

        e.transmittance = (1 - e.reflectance)/(distance*50);
        return e;  }
};




class LightSource
{
public:
    virtual float3 getPowerDensityAt ( float3 x )=0;
    virtual float3 getLightDirAt     ( float3 x )=0;
    virtual float  getDistanceFrom   ( float3 x )=0;
};

class DirectionalLight : public LightSource{
    float3 powerDensity;
    float3 lightDir;

    public : DirectionalLight(float3 powerDensity, float3 Direction): powerDensity(powerDensity),lightDir(Direction){}
    float3 getPowerDensityAt ( float3 x ){
        return powerDensity;
        
    }
    float3 getLightDirAt     ( float3 x ){
        return lightDir;
    }
    float  getDistanceFrom   ( float3 x ){
        return MAXFLOAT;
    }
};

class PointLight : public LightSource{
float3 powerDensity;
float3 position;

public : PointLight(float3 powerDensity, float3 position): powerDensity(powerDensity),position(position){}

    

float3 getLightDirAt     ( float3 x ){
    float3 lightDirectionVector = position-x;
    return lightDirectionVector.normalize();
}
float  getDistanceFrom   ( float3 x ){
    float3 lightDirectionVector = position-x;
    return lightDirectionVector.norm();
}
float3 getPowerDensityAt ( float3 x ){
    float term = 1/(4 * M_PI * getDistanceFrom(x) * getDistanceFrom(x));
    return powerDensity*term;
    }
};

class ClippedQuadric : public Intersectable
{
    Quadric shape;
    Quadric clipper;
    
public:
    ClippedQuadric(Material* material):
    Intersectable(material), shape(material, float4x4()), clipper(material, float4x4())
    {
        shape.sphere();
        clipper.parallelPlanes();
    }

    Hit intersect(const Ray& ray)
    {
        // This is a generic intersect that works for any shape with a quadratic equation. solveQuadratic should solve the proper equation (+ ray equation) for the shape, and getNormalAt should return the proper normal
        
        
        QuadraticRoots roots  = shape.solveQuadratic(ray);
        float3 pOne = ray.origin+ray.dir*roots.t1;
        float3 pTwo = ray.origin+ray.dir*roots.t2;
        if (clipper.contains(pOne))
        {
            roots.t1 = -1;
        }

        if(clipper.contains(pTwo)){

            roots.t2 = -1;
        }
     
        
        //getlesserposisitve on these roots and the rest is the same.
        float t = roots.getLesserPositive();
        Hit hit;
        hit.t = t;
        hit.material = material;
        hit.position = ray.origin + ray.dir * t;
        hit.normal = shape.getNormalAt(hit.position);
        return hit;
    }

    ClippedQuadric* parallelPlanes(){
        shape.parallelPlanes();
        clipper.parallelPlanes();
        return this;
    };
    
    
    ClippedQuadric* cone(){
        shape.cone()->transform(float4x4::scaling(float3(.5, 1, .5)));
        clipper.parallelPlanes()->transform(float4x4::translation(float3(0,-2,0)));
        return this;
    };
    
    ClippedQuadric* sphere(){
        shape.sphere();
        clipper.parallelPlanes();
        return this;
    };
    
    ClippedQuadric* cylinder(){
        shape.cylinder()->transform(float4x4::scaling(float3(.25, .51, .25)));
        clipper.parallelPlanes()->transform(float4x4::translation(float3(0,-0.4,0)));
        return this;
    };
    
    ClippedQuadric* hyperboloid(){
        shape.hyperboloid()->transform(float4x4::scaling(float3(.25, .7, .25)));
        clipper.parallelPlanes()->transform(float4x4::translation(float3(0,-0.4,0)));
        return this;
    };
    ClippedQuadric* bishopBody(){
        shape.hyperboloid()->transform(float4x4::scaling(float3(.19, .7, .19)));
        clipper.parallelPlanes()->transform(float4x4::translation(float3(0,-0.4,0)));
        return this;
    };
    ClippedQuadric* kingBody(){
        shape.hyperboloid()->transform(float4x4::scaling(float3(.25, .5, .25)));
        clipper.parallelPlanes()->transform(float4x4::translation(float3(0,-0.3,0)));
        return this;
    };
    ClippedQuadric* hyperbolicCyl(){
        shape.hyperbolicCyl();
        clipper.parallelPlanes();
        return this;
    };
    
    ClippedQuadric* hyperbolicPara(){
        shape.hyperbolicPara();
        clipper.parallelPlanes();
        return this;
    };
    
    ClippedQuadric* parabaloid(){
        shape.paraboloid();
        clipper.parallelPlanes();
        return this;
    };
    
    ClippedQuadric* transform(float4x4 t){
        shape.transform(t);
        clipper.transform(t);
        return this;
    };
};

class Board : public Intersectable
{
    float3 ro;
    float3 n;
    float3 bottomCorner;
    Material* materialOne;
    Material* materialTwo;
    Material* wood = new Wood();
    
public:
    Board(const float3& point, float3 normal, Material* materialOne, Material* materialTwo, float3 bottonCorner):
    Intersectable(materialOne),materialOne(materialOne),materialTwo(materialTwo),ro(point), n(normal),bottomCorner(bottonCorner)
    {}
    
    Hit intersect(const Ray& ray){
        Hit hit;
        float3 topCorner = bottomCorner+float3(8,0,8);
        float numerator = (ro-ray.origin).dot(n);
        float denominator = ray.dir.dot(n);
        hit.t=numerator/denominator;
        hit.position = ray.origin + ray.dir * hit.t;
        hit.normal = n;
        if((hit.position.x>bottomCorner.x&&hit.position.z>bottomCorner.z)&&(hit.position.x<topCorner.x&&hit.position.z<topCorner.z)){
            if(((((int)(floorf(hit.position.x))%2)) == 0 && (((int)(floorf(hit.position.z))%2)) == 0)||((((int)(floorf(hit.position.x))%2)) != 0 && (((int)(floorf(hit.position.z))%2)) != 0)){
                hit.material = materialOne;
                return hit;
            }
            hit.material = materialTwo;
            return hit;
        }
       if((hit.position.x<topCorner.x+2&&hit.position.z<topCorner.z+1)){
        hit.material = wood;
        return hit;
       }
        hit.t=-1;
        return hit;
    }
};

class FrontWall : public Intersectable
{
    float3 ro;
    float3 n;
    float3 topCorner;
    Material* materialOne;
    Material* materialTwo;
    
public:
    FrontWall(const float3& point, float3 normal, Material* materialOne, Material* materialTwo, float3 topCorner):
    Intersectable(materialOne),materialOne(materialOne),materialTwo(materialTwo),ro(point), n(normal),topCorner(topCorner)
    {}
    
    Hit intersect(const Ray& ray){
        Hit hit;
        float3 bottonCorner = topCorner+float3(8,0,8);
        float numerator = (ro-ray.origin).dot(n);
        float denominator = ray.dir.dot(n);
        hit.t=numerator/denominator;
        hit.position = ray.origin + ray.dir * hit.t;
        hit.normal = n;
        if((hit.position.x>topCorner.x&&hit.position.y<topCorner.y)&&(hit.position.x<bottonCorner.x&&hit.position.y<bottonCorner.y)){
            if(((((int)(floorf(hit.position.x))%2)) == 0 && (((int)(floorf(hit.position.z))%2)) == 0)||((((int)(floorf(hit.position.x))%2)) == 0)){
                hit.material = materialOne;
                return hit;
            }
            hit.material = materialTwo;
            return hit;
        }
        hit.t=-1;
            return hit;
        }
};
class multiQuadric : public Intersectable
{
    std::vector<Intersectable*> chess;
public:
    
    multiQuadric(Material* material):
    Intersectable(material)
    {
    }
    
    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        float t = MAXFLOAT;
        for(int i = 0; i<chess.size(); i++) {
            Hit h = chess.at(i)->intersect(ray);
            if (h.t < t && h.t>0) {
                bestHit = h;
                t = h.t;
            }
        }
       return bestHit;
    }
    Hit intersect(const Ray& ray)
    {
        Hit bestHit = firstIntersect(ray);
        return bestHit;
    }
    
    multiQuadric* transform(float4x4 t){
        for(unsigned int i=0; i<chess.size();i++){
            if(Quadric* quadric = dynamic_cast<Quadric*>(chess.at(i))){
                quadric->transform(t);
            }
            if(ClippedQuadric* clippedquadric = dynamic_cast<ClippedQuadric*>(chess.at(i))){
                clippedquadric->transform(t);
            }
        }
               return this;
    };
    
    multiQuadric* Pawn(){
        
        ClippedQuadric* sphere = (new ClippedQuadric(material))->sphere()->transform(float4x4::rotation(float3(1,1,1), .5))->transform(float4x4::scaling(float3(.25, .25, .25)));
        ClippedQuadric* cone = (new ClippedQuadric(material ))->cone()->transform(float4x4::rotation(float3(1,0,0), .5))->transform(float4x4::scaling(float3(.25, .25, .25)));
        chess.push_back(sphere);
        chess.push_back(cone);
        return this;
    };
    
    multiQuadric* Queen(){

        ClippedQuadric* hyperbol = (new ClippedQuadric(material))->hyperboloid()->transform(float4x4::scaling(float3(.001, .005, 0)))->transform(float4x4::translation(float3( -.5,.15,-1.47)));
        ClippedQuadric* sphere = (new ClippedQuadric(material))->sphere()->transform(float4x4::scaling(float3(.09,.09,.09)))->transform(float4x4::translation(float3( -.5,1.1,-1.51)));
        ClippedQuadric* ringOne = (new ClippedQuadric(material))->sphere()->transform(float4x4::scaling(float3(.39,.09,.09)))->transform(float4x4::translation(float3( -.5,.79,-1.44)));
         ClippedQuadric* ringTwo = (new ClippedQuadric(material))->sphere()->transform(float4x4::scaling(float3(.3,.04,.04)))->transform(float4x4::translation(float3( -.5,.19,-1.27)));
         ClippedQuadric* ringThree = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.45))->transform(float4x4::scaling(float3(.40,.05,.13)))->transform(float4x4::translation(float3( -.5,-.54,-1.08)));
        ClippedQuadric* dome = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.45))->transform(float4x4::scaling(float3(.29,.29,.29)))->transform(float4x4::translation(float3( -.5,.76,-1.3)));
        chess.push_back(dome);
        chess.push_back(sphere);
        chess.push_back(ringOne);
        chess.push_back(ringTwo);
        //chess.push_back(ringThree);
        chess.push_back(hyperbol);
        
        return this;
    };
    multiQuadric* Bishop(){
        
        ClippedQuadric* hyperbol = (new ClippedQuadric(material))->bishopBody()->transform(float4x4::scaling(float3(.0001, .5, 0)))->transform(float4x4::translation(float3( -.5,-.15,-1.47)));
        
        ClippedQuadric* sphere = (new ClippedQuadric(material))->sphere()->transform(float4x4::scaling(float3(.06,.07,.06)))->transform(float4x4::translation(float3( -.5,.91,-1.4)));
        
        ClippedQuadric* ringThree = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.45))->transform(float4x4::scaling(float3(.40,.05,.13)))->transform(float4x4::translation(float3( -.5,-.54,-1.08)));
        
        ClippedQuadric* dome = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.45))->transform(float4x4::scaling(float3(.25,.39,.25)))->transform(float4x4::translation(float3( -.5,.46,-1.16)));
        
        chess.push_back(dome);
        chess.push_back(sphere);
        chess.push_back(hyperbol);
        return this;
    };
    
    
    multiQuadric* King(){
        
        ClippedQuadric* hyperbol = (new ClippedQuadric(material))->kingBody()->transform(float4x4::scaling(float3(.001, .005, 0)))->transform(float4x4::translation(float3( -.5,.19,-1.47)));
        ClippedQuadric* crossone = (new ClippedQuadric(material))->hyperboloid()->transform(float4x4::scaling(float3(.19,.19,.19)))->transform(float4x4::translation(float3( -.5,1.2,-1.51)));
        ClippedQuadric* crosstwo = (new ClippedQuadric(material))->hyperboloid()->transform(float4x4::rotation(float3(0,0,1), 1))->transform(float4x4::scaling(float3(.19,.09,.19)))->transform(float4x4::translation(float3( -.5,1.2,-1.51)));
        ClippedQuadric* ringOne = (new ClippedQuadric(material))->sphere()->transform(float4x4::scaling(float3(.39,.09,.09)))->transform(float4x4::translation(float3( -.5,.79,-1.44)));
        ClippedQuadric* ringThree = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.45))->transform(float4x4::scaling(float3(.40,.05,.13)))->transform(float4x4::translation(float3( -.5,-.54,-1.08)));
        ClippedQuadric* dome = (new ClippedQuadric(material))->parabaloid()->transform(float4x4::rotation(float3(1,0,0), -1.5))->transform(float4x4::scaling(float3(.31,.35,.31)))->transform(float4x4::translation(float3( -.48,.79,-1.22)));
        chess.push_back(dome);
        chess.push_back(crossone);
        chess.push_back(crosstwo);
        chess.push_back(ringOne);
        //chess.push_back(ringThree);
        chess.push_back(hyperbol);
        
        return this;
    };
    multiQuadric* Rook(){
        ClippedQuadric* hyperbol = (new ClippedQuadric(material))->cylinder()->transform(float4x4::translation(float3( -2.5,.19,-1.47)));
        chess.push_back(hyperbol);
        return this;
    };
    
    
};


class Scene
{
    Camera camera;
    std::vector<Intersectable*> objects;
    std::vector<Material*> materials;
    std::vector<LightSource*> lights;
public:
    Scene()
    {

//0
        materials.push_back(new PhongBlinn(float3(.75,.75,.75), 2 ));
//1
        materials.push_back(new BumpyMetal(goldRI,goldEC,20));
//2
        materials.push_back(new BumpyMetal(silverRI,silverEC,20));
//3
        materials.push_back(new Dielectric(.9));
//4
        materials.push_back(new Wood());
//5
        materials.push_back(new Marble());
//6
        materials.push_back(new Diffuse(float3(1,1,1),10));
//7
        materials.push_back(new Diffuse(float3(0,0,0),10));
//8
        materials.push_back(new Plastic(float3(.85,.75,.75), 3 ));
//9
        materials.push_back(new Diffuse(float3(.3,.3,.3),14));
//10 glass
        materials.push_back(new Dielectric(1.4));
 //11 silver
        materials.push_back(new Metal(silverRI,silverEC));
 //12 gold
        materials.push_back(new Metal(goldRI,goldEC));
//13 gemstone
        materials.push_back(new Gemstone(.4));
      // transform then scale then rotate then translate!!

        Plane *backWall = new Plane(float3(0,1,-15),float3(0,0,1), materials.at(9));
        Plane *leftWall = new Plane(float3(-8,0,1),float3(1,0,0), materials.at(9));
        Plane *rightWall = new Plane(float3(8,0,1),float3(1,0,0), materials.at(9));
        FrontWall *frontWall = new FrontWall(float3(0,0,-.08),float3(0,0,1), materials.at(5), materials.at(8),float3(-4,-0.99,2));
        Board *board = new Board(float3(0,-1,0),float3(0,1,.3), materials.at(6), materials.at(9),float3(-4,-2,-9));
        
        objects.push_back(board);
        objects.push_back(backWall);
        objects.push_back(leftWall);
        objects.push_back(rightWall);
        objects.push_back(frontWall);
//      Set Up Area
        
        Quadric* sphere = (new Quadric(materials.at(10)))->sphere()->transform(float4x4::scaling(float3(1, 1, 1))*float4x4::translation(float3(0,3,-5)));
        objects.push_back(sphere);

//      Pawns
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3( 2.9,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3( 2.1,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3( 1.3,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3( 0.4,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3(-0.2,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3(-1.2,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3(-2.1,.29,-1.5))));
        objects.push_back((new multiQuadric(materials.at(8)))->Pawn()->transform(float4x4::scaling(float3(.6, .6, .6)))->transform(float4x4::translation(float3(-2.9,.29,-1.5))));

//      Queen
         objects.push_back((new multiQuadric(materials.at(12)))->Queen()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(0,0,0))));
  
//      Bishops
        objects.push_back((new multiQuadric(materials.at(1)))->Bishop()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(-1,0,0))));
        objects.push_back((new multiQuadric(materials.at(2)))->Bishop()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(2,0,0))));

//      King
        objects.push_back((new multiQuadric(materials.at(10)))->King()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(1,0,0))));

//      Rooks
        objects.push_back((new multiQuadric(materials.at(13)))->Rook()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(-1,0,0))));
        objects.push_back((new multiQuadric(materials.at(13)))->Rook()->transform(float4x4::scaling(float3(1, 1, 1)))->transform(float4x4::translation(float3(6,0,0))));
        
       DirectionalLight *light = new DirectionalLight(float3(10,10,10),float3(1,0,1));
     // lights.push_back(light);
        PointLight *pointLight  = new PointLight(float3(500,500,500),float3(3,7,-8));
       lights.push_back(pointLight);
        PointLight *pointLighto  = new PointLight(float3(500,700,500),float3(0,7.5,-5));
        lights.push_back(pointLighto);
       PointLight *pointLighta  = new PointLight(float3(500,500,500),float3(1,2,6));
        lights.push_back(pointLighta);
    }
    ~Scene()
    {
        for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
        	delete *iMaterial;
        for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
        	delete *iObject;
    }
    
public:
    Camera& getCamera()
    {
        return camera;
    }
    Hit firstIntersect(Ray ray) {
        
        Hit bestHit;
        float t = MAXFLOAT;
        for(int i = 0; i<objects.size(); i++) {
            Hit h = objects.at(i)->intersect(ray);
            if (h.t < t && h.t>0) {
                bestHit = h;
                t = h.t;
            }
        }
        return bestHit;
    }
    
    float3 trace(const Ray& ray, int depth)
    {
        
        if(depth > maxDepth){
            return float3(1,1,1);
        }
        float3 outRadiance;
        Hit bestHit = firstIntersect(ray);
        if(bestHit.t < 0){ return float3(1,1,1);}
        float eps = .01;
        for(unsigned int i = 0; i<lights.size(); i++){
            
            Ray shadowRay(bestHit.position + (bestHit.normal*eps), lights.at(i)->getLightDirAt(bestHit.position));
            Hit shadowHit = firstIntersect(shadowRay);
            
            if(shadowHit.t > 0 && shadowHit.t < lights.at(i)->getDistanceFrom(bestHit.position)) continue;
            outRadiance +=  bestHit.material->shade(bestHit.normal, -ray.dir,
                                                    lights.at(i)->getLightDirAt(bestHit.position),
                                                    lights.at(i)->getPowerDensityAt(bestHit.position), bestHit.position);
        }

        if(BumpyMetal* bumpmetal = dynamic_cast<BumpyMetal*>(bestHit.material)){
            BumpyMetal::Event e = bumpmetal->evaluateEvent(ray.dir, bestHit.normal,bestHit.position);
            outRadiance += trace( Ray(bestHit.position + (bestHit.normal*eps), e.reflectionDir), depth+1 ) * e.reflectance;
            
        }
        if(Metal* metal = dynamic_cast<Metal*>(bestHit.material)){
            Metal::Event e = metal->evaluateEvent(ray.dir, bestHit.normal);
            outRadiance += trace( Ray(bestHit.position + (bestHit.normal*eps), e.reflectionDir), depth+1 ) * e.reflectance;
                    }
        if(Dielectric* dielectric =
           dynamic_cast<Dielectric*>(bestHit.material)){
            Dielectric::Event e = dielectric->evaluateEvent(ray.dir, bestHit.normal);
            outRadiance += trace( Ray(bestHit.position + (bestHit.normal*eps), e.reflectionDir), depth+1) * e.reflectance;
            if(e.transmittance > 0)
                outRadiance += trace( Ray(bestHit.position - (bestHit.normal* eps), e.refractionDir), depth+1) * e.transmittance ;
        }
        if(Gemstone* gemstone =
           dynamic_cast<Gemstone*>(bestHit.material)){
            Gemstone::Event e = gemstone->evaluateEvent(ray.dir, bestHit.normal,bestHit.t);
            outRadiance += trace( Ray(bestHit.position + (bestHit.normal*eps), e.reflectionDir), depth+1) * e.reflectance;
            if(e.transmittance > 0)
                outRadiance += trace( Ray(bestHit.position - (bestHit.normal* eps), e.refractionDir), depth+1) * e.transmittance ;
        }
        for(int i=0; i<objects.size(); i++){
        Hit hit = objects.at(i)->intersect(ray);
            if((hit.t<bestHit.t)&&(hit.t>0)){
                bestHit = hit;
            }
        }
        if(bestHit.t == MAXFLOAT){
            return float3(1,1,1);
        }
               return outRadiance;
        
}
};



////////////////////////////////////////////////////////////////////////////////////////////////////////
// global application data

// screen resolution
const int screenWidth = 600;
const int screenHeight = 600;
// image to be computed by ray tracing
float3 image[screenWidth*screenHeight];

Scene scene;

bool computeImage()
{
    static unsigned int iPart = 0;
    
    if(iPart >= 64)
        return false;
    for(int j = iPart; j < screenHeight; j+=64)
    {
        for(int i = 0; i < screenWidth; i++)
        {
            float3 pixelColor = float3(0, 0, 0);
            float2 ndcPixelCentre( (2.0 * i - screenWidth) / screenWidth, (2.0 * j - screenHeight) / screenHeight );
            
            Camera& camera = scene.getCamera();
            Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcPixelCentre));
            
            image[j*screenWidth + i] = scene.trace(ray, 0);
        }
    }
    iPart++;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL starts here. OpenGL just outputs the image computed to the array.

// display callback invoked when window needs to be redrawn
void onDisplay( ) {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen
    
    if(computeImage())
        glutPostRedisplay();
    glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
    
    glutSwapBuffers(); // drawing finished
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);						// initialize GLUT
    glutInitWindowSize(screenWidth, screenHeight);				// startup window size 
    glutInitWindowPosition(100, 100);           // where to put window on screen
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);    // 8 bit R,G,B,A + double buffer + depth buffer
    
    glutCreateWindow("Ray caster");				// application window is created and displayed
    
    glViewport(0, 0, screenWidth, screenHeight);
    
    glutDisplayFunc(onDisplay);					// register callback
    
    glutMainLoop();								// launch event handling loop
    
    return 0;
}

