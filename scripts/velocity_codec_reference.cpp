// Independent numerical fixture for the documented velocity codec, no SDK or robot I/O.
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
  int count;
  if (!(std::cin >> count) || count < 1 || count > 300) return 2;
  std::array<float,29> defaults{}, scale{}, offset{}, last{};
  for (auto* a : {&defaults,&scale,&offset}) for (float& x:*a) if (!(std::cin>>x)) return 3;
  const float dt=.02f,period=.6f;
  const float delta=dt*(1.f/period);
  float phase=delta; // reset() primes observation terms once, without a policy call.
  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (int frame=0;frame<count;++frame) {
    std::array<float,4> q{};
    std::array<float,3> gyro{},cmd{};
    std::array<float,29> pos{},vel{},action{};
    for (float& x:q) if (!(std::cin>>x)) return 4;
    for (float& x:gyro) if (!(std::cin>>x)) return 5;
    for (auto* a : {&pos,&vel}) for (float& x:*a) if (!(std::cin>>x)) return 6;
    for (float& x:cmd) if (!(std::cin>>x)) return 7;
    for (float& x:action) if (!(std::cin>>x)) return 8;
    const Eigen::Quaternionf quat(q[0],q[1],q[2],q[3]);
    const Eigen::Vector3f gravity=quat.conjugate()*Eigen::Vector3f(0.f,0.f,-1.f);
    phase=std::fmod(phase+delta,1.f);
    const float norm=std::sqrt(cmd[0]*cmd[0]+cmd[1]*cmd[1]+cmd[2]*cmd[2]);
    const float sine=norm<.1f ? 0.f : std::sin(phase*2*M_PI);
    const float cosine=norm<.1f ? 0.f : std::cos(phase*2*M_PI);
    for (float x:gyro) std::cout<<x<<' ';
    for (int i=0;i<3;++i) std::cout<<gravity[i]<<' ';
    for (float x:cmd) std::cout<<x<<' ';
    std::cout<<sine<<' '<<cosine<<' ';
    for (int i=0;i<29;++i) std::cout<<(pos[i]-defaults[i])<<' ';
    for (float x:vel) std::cout<<x<<' ';
    for (float x:last) std::cout<<x<<' ';
    for (int i=0;i<29;++i) {
      float target=action[i]*scale[i];target+=offset[i];
      std::cout<<target<<(i==28 ? '\n' : ' ');
    }
    last=action;
  }
  return 0;
}
