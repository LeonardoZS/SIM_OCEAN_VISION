import holoocean
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#pip install icecream
#pip install matplotlib
#pip install numpy

RangeMaxSS = 50
RangeBinsSS = 2000


cfg = {
    "name": "TorpedoSidescanSonar",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "frames_per_sec": True,
    "window_width": 1280,
    "window_height": 720,
    "octree_max": 5.0,
    "octree_min": 0.02,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "control_scheme": 0,
            "location": [203, 457, -284], 
            "rotation": [0, 0, 180],
            "sensors": [
                {"sensor_type": "PoseSensor", "socket": "IMUSocket"},
                {"sensor_type": "VelocitySensor", "socket": "IMUSocket"},
                {
                    "sensor_type": "IMUSensor",
                    "socket": "IMUSocket",
                    "Hz": 10,
                    "configuration": {
                        "ReturnBias": True,
                        "AccelSigma": 0.00277,
                        "AccelBiasSigma": 0.00141,
                        "AngVelSigma": 0.00123,
                        "AngVelBiasSigma": 0.00388
                    }
                },
                {
                    "sensor_type": "GPSSensor",
                    "socket": "IMUSocket",
                    "Hz": 5,
                    "configuration": {
                        "Depth": 1,
                        "Sigma": 0.5,
                        "DepthSigma": 0.25
                    }
                },
                {
                    "sensor_type": "DVLSensor",
                    "socket": "DVLSocket",
                    "Hz": 10,
                    "configuration": {
                        "RangeSigma": 0.1,
                        "ReturnRange": True,
                        "VelSigma": 0.02626,
                        "Elevation": 22.5,
                        "MaxRange": 50
                    }
                },
                {
                    "sensor_type": "DepthSensor",
                    "socket": "DepthSocket",
                    "Hz": 10,
                    "configuration": {
                        "Sigma": 0.255
                    }
                },
                {
                    "sensor_type": "SidescanSonar",
                    "socket": "SonarSocket",
                    "Hz": 10,
                    "location": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "configuration": {
                        "RangeMin": 0.5,
                        "RangeMax": RangeMaxSS,
                        "RangeBins": RangeBinsSS,
                        "Azimuth": 170,
                        "InitOctreeRange": 50,
                    }
                },
                {
                    "sensor_type": "CameraSensor",
                    "socket": "CameraSocket",
                    "Hz": 10,
                    "location": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "width": 640,
                    "height": 480,
                    "fov": 90
                }
            ]
        }
    ]
}

ic(cfg)


plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))


#================================
#SONAR SIDE SCAN (TOPO-ESQUERDA)
#================================
t = np.arange(0, 80)
r = np.linspace(-RangeMaxSS, RangeMaxSS, RangeBinsSS)
R, T = np.meshgrid(r, t)
dados_sonar_side_scan = np.zeros_like(R)

plot_sonar = axs[0, 0].pcolormesh(R, T, dados_sonar_side_scan, cmap='copper', shading='auto', vmin=0, vmax=1)
axs[0, 0].invert_yaxis()
axs[0, 0].set_title("Sidescan Sonar")
axs[0, 0].set_xlabel("Distância (m)")
axs[0, 0].set_ylabel("Tempo")
axs[0, 0].grid(False)

#================================
#CAMERA (TOPO-DIREITA)
#================================
plot_camera = axs[0, 1].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
axs[0, 1].set_title("Visão da Camera")
axs[0, 1].axis("off")

#================================
#PROFUNIDADE (TOPO-DIREITA)
#================================
profundidades = []
tempo = []
plot_profundidade, = axs[1, 0].plot([], [], color='blue')
axs[1, 0].set_title("Profundidade vs Tempo")
axs[1, 0].set_xlabel("Passo de Tempo")
axs[1, 0].set_ylabel("Profundidade (m)")
axs[1, 0].invert_yaxis()


plt.tight_layout()
plt.gcf().canvas.flush_events()

#### ENV SETUP
env = holoocean.make(scenario_cfg=cfg)
env.reset()
env.act('auv0', np.array([0, 0, 0, 0, 25]))


for i in range(3000):

    state = env.tick()

    if 'SidescanSonar' in state:
        dados_sonar_side_scan = np.roll(dados_sonar_side_scan, 1, axis=0)
        dados_sonar_side_scan[0] = state['SidescanSonar']
        plot_sonar.set_array(dados_sonar_side_scan.ravel())


    if "CameraSensor" in state:
        imagem_atual = state["CameraSensor"]
        plot_camera.set_data(imagem_atual)

    if "DepthSensor" in state:
        profundidade = state["DepthSensor"]
        profundidades.append(profundidade)
        tempo.append(i)
        plot_profundidade.set_data(tempo, profundidades)
        axs[1, 0].set_xlim(0, max(100, i))
        axs[1, 0].set_ylim(max(profundidades)+5, min(profundidades)-5)

    plt.draw()
    plt.pause(0.001)

print("Simulação encerrada!")
plt.ioff()
plt.show()
