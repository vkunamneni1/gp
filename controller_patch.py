--- controller.py	2026-06-02
+++ controller.py	2026-06-02
@@ -216,6 +216,7 @@
         # Apply the throttling to our lateral/forward velocity
         vel_cmd[0] *= alignment_factor
         vel_cmd[1] *= alignment_factor
+        vel_cmd[2] *= alignment_factor
 
         self._send_velocity_ned(vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate)
 
