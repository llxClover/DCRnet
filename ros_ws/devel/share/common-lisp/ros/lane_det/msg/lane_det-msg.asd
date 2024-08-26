
(cl:in-package :asdf)

(defsystem "lane_det-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Localization" :depends-on ("_package_Localization"))
    (:file "_package_Localization" :depends-on ("_package"))
    (:file "lane_detection" :depends-on ("_package_lane_detection"))
    (:file "_package_lane_detection" :depends-on ("_package"))
  ))