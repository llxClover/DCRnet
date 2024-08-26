;; Auto-generated. Do not edit!


(when (boundp 'lane_det::lane_detection)
  (if (not (find-package "LANE_DET"))
    (make-package "LANE_DET"))
  (shadow 'lane_detection (find-package "LANE_DET")))
(unless (find-package "LANE_DET::LANE_DETECTION")
  (make-package "LANE_DET::LANE_DETECTION"))

(in-package "ROS")
;;//! \htmlinclude lane_detection.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass lane_det::lane_detection
  :super ros::object
  :slots (_left_1 _left_0 _right_0 _right_1 _coeff_left_1 _coeff_left_0 _coeff_right_0 _coeff_right_1 ))

(defmethod lane_det::lane_detection
  (:init
   (&key
    ((:left_1 __left_1) ())
    ((:left_0 __left_0) ())
    ((:right_0 __right_0) ())
    ((:right_1 __right_1) ())
    ((:coeff_left_1 __coeff_left_1) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:coeff_left_0 __coeff_left_0) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:coeff_right_0 __coeff_right_0) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:coeff_right_1 __coeff_right_1) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _left_1 __left_1)
   (setq _left_0 __left_0)
   (setq _right_0 __right_0)
   (setq _right_1 __right_1)
   (setq _coeff_left_1 __coeff_left_1)
   (setq _coeff_left_0 __coeff_left_0)
   (setq _coeff_right_0 __coeff_right_0)
   (setq _coeff_right_1 __coeff_right_1)
   self)
  (:left_1
   (&rest __left_1)
   (if (keywordp (car __left_1))
       (send* _left_1 __left_1)
     (progn
       (if __left_1 (setq _left_1 (car __left_1)))
       _left_1)))
  (:left_0
   (&rest __left_0)
   (if (keywordp (car __left_0))
       (send* _left_0 __left_0)
     (progn
       (if __left_0 (setq _left_0 (car __left_0)))
       _left_0)))
  (:right_0
   (&rest __right_0)
   (if (keywordp (car __right_0))
       (send* _right_0 __right_0)
     (progn
       (if __right_0 (setq _right_0 (car __right_0)))
       _right_0)))
  (:right_1
   (&rest __right_1)
   (if (keywordp (car __right_1))
       (send* _right_1 __right_1)
     (progn
       (if __right_1 (setq _right_1 (car __right_1)))
       _right_1)))
  (:coeff_left_1
   (&optional __coeff_left_1)
   (if __coeff_left_1 (setq _coeff_left_1 __coeff_left_1)) _coeff_left_1)
  (:coeff_left_0
   (&optional __coeff_left_0)
   (if __coeff_left_0 (setq _coeff_left_0 __coeff_left_0)) _coeff_left_0)
  (:coeff_right_0
   (&optional __coeff_right_0)
   (if __coeff_right_0 (setq _coeff_right_0 __coeff_right_0)) _coeff_right_0)
  (:coeff_right_1
   (&optional __coeff_right_1)
   (if __coeff_right_1 (setq _coeff_right_1 __coeff_right_1)) _coeff_right_1)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point[] _left_1
    (apply #'+ (send-all _left_1 :serialization-length)) 4
    ;; geometry_msgs/Point[] _left_0
    (apply #'+ (send-all _left_0 :serialization-length)) 4
    ;; geometry_msgs/Point[] _right_0
    (apply #'+ (send-all _right_0 :serialization-length)) 4
    ;; geometry_msgs/Point[] _right_1
    (apply #'+ (send-all _right_1 :serialization-length)) 4
    ;; float32[] _coeff_left_1
    (* 4    (length _coeff_left_1)) 4
    ;; float32[] _coeff_left_0
    (* 4    (length _coeff_left_0)) 4
    ;; float32[] _coeff_right_0
    (* 4    (length _coeff_right_0)) 4
    ;; float32[] _coeff_right_1
    (* 4    (length _coeff_right_1)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point[] _left_1
     (write-long (length _left_1) s)
     (dolist (elem _left_1)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Point[] _left_0
     (write-long (length _left_0) s)
     (dolist (elem _left_0)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Point[] _right_0
     (write-long (length _right_0) s)
     (dolist (elem _right_0)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Point[] _right_1
     (write-long (length _right_1) s)
     (dolist (elem _right_1)
       (send elem :serialize s)
       )
     ;; float32[] _coeff_left_1
     (write-long (length _coeff_left_1) s)
     (dotimes (i (length _coeff_left_1))
       (sys::poke (elt _coeff_left_1 i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _coeff_left_0
     (write-long (length _coeff_left_0) s)
     (dotimes (i (length _coeff_left_0))
       (sys::poke (elt _coeff_left_0 i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _coeff_right_0
     (write-long (length _coeff_right_0) s)
     (dotimes (i (length _coeff_right_0))
       (sys::poke (elt _coeff_right_0 i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[] _coeff_right_1
     (write-long (length _coeff_right_1) s)
     (dotimes (i (length _coeff_right_1))
       (sys::poke (elt _coeff_right_1 i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point[] _left_1
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _left_1 (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _left_1)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Point[] _left_0
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _left_0 (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _left_0)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Point[] _right_0
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _right_0 (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _right_0)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Point[] _right_1
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _right_1 (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _right_1)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float32[] _coeff_left_1
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _coeff_left_1 (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _coeff_left_1 i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _coeff_left_0
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _coeff_left_0 (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _coeff_left_0 i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _coeff_right_0
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _coeff_right_0 (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _coeff_right_0 i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;; float32[] _coeff_right_1
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _coeff_right_1 (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _coeff_right_1 i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get lane_det::lane_detection :md5sum-) "6a8ccbd1be031a9ebcd2bcf06c6e28fd")
(setf (get lane_det::lane_detection :datatype-) "lane_det/lane_detection")
(setf (get lane_det::lane_detection :definition-)
      "geometry_msgs/Point[] left_1
geometry_msgs/Point[] left_0
geometry_msgs/Point[] right_0
geometry_msgs/Point[] right_1
float32[] coeff_left_1
float32[] coeff_left_0
float32[] coeff_right_0
float32[] coeff_right_1

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :lane_det/lane_detection "6a8ccbd1be031a9ebcd2bcf06c6e28fd")


