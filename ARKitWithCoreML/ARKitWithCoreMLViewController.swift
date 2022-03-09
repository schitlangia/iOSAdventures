//
//  ARKitWithCoreMLViewController.swift
//  ARKitWithCoreML
//
//
//  Created by Chitlangia, Shweta on 09/03/22.
//  This class uses CoreML with ARKit to classify objects identified in camera on tap.
//  Bring the object close to the camera and start tapping to classify it.

import UIKit
import SceneKit
import ARKit

class ARKitWithCoreMLViewController: UIViewController, ARSCNViewDelegate {
    
    // MARK: - IBOutlets
    @IBOutlet var sceneView: ARSCNView!
    
    // MARK: - Private vars
    private var resnetModel: Resnet50?
    private var rayCastResult: ARRaycastResult?
    private var visionRequests = [VNRequest]()
    
    // MARK: - View Lifecycle methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize the Resnet ML model
        do {
            resnetModel = try Resnet50(configuration: MLModelConfiguration())
        } catch {
            resnetModel = nil
        }
        
        registerGestureRecognizers()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    // MARK: - Private functions
    
    private func registerGestureRecognizers() {
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(tapped))
        sceneView.addGestureRecognizer(tapGestureRecognizer)
    }
    
    @objc func tapped(recognizer :UIGestureRecognizer) {
        let touchLocation = recognizer.location(in: sceneView)
        rayCastingMethod(touchLocation: touchLocation)
        
        if let rayCastResult = rayCastResult {
            guard let currentFrame = sceneView.session.currentFrame else {
                print("Invalid frame")
                return
            }
            let pixelBuffer = currentFrame.capturedImage
            performVisionRequest(pixelBuffer: pixelBuffer, rayCastResult: rayCastResult)
        }
    }
    
    func rayCastingMethod(touchLocation: CGPoint) {
        // Ray's target is a plane that is estimated using the feature points around the ray.
        guard let raycastQuery = sceneView.raycastQuery(from: touchLocation,
                                                        allowing: .estimatedPlane,
                                                        alignment: .any) else {
            print("Invalid raycastQuery")
            return
        }
        
        
        guard let firstRayCastResult = sceneView.session.raycast(raycastQuery).first else {
            print("No results found for raycasting")
            return
        }
        
        rayCastResult = firstRayCastResult
    }
    
    
    private func displayPredictions(text :String, rayCastResult: ARRaycastResult) {
        let node = createText(text: text)
        node.position = SCNVector3(rayCastResult.worldTransform.columns.3.x,
                                   rayCastResult.worldTransform.columns.3.y,
                                   rayCastResult.worldTransform.columns.3.z)
        sceneView.scene.rootNode.addChildNode(node)
    }
    
    private func createText(text: String) -> SCNNode {
        
        let parentNode = SCNNode()
        
        let sphere = SCNSphere(radius: 0.01)
        let sphereMaterial = SCNMaterial()
        sphereMaterial.diffuse.contents = UIColor.random
        sphere.firstMaterial = sphereMaterial
        let sphereNode = SCNNode(geometry: sphere)
        
        let textGeometry = SCNText(string: text, extrusionDepth: 0)
        textGeometry.alignmentMode = CATextLayerAlignmentMode.center.rawValue
        textGeometry.firstMaterial?.diffuse.contents = UIColor.random
        textGeometry.firstMaterial?.specular.contents = UIColor.white
        textGeometry.firstMaterial?.isDoubleSided = true
        
        let font = UIFont(name: "Futura", size: 0.15)
        textGeometry.font = font
        let textNode = SCNNode(geometry: textGeometry)
        textNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        parentNode.addChildNode(sphereNode)
        parentNode.addChildNode(textNode)
        return parentNode
    }
    
    private func performVisionRequest(pixelBuffer :CVPixelBuffer, rayCastResult: ARRaycastResult) {
        
        guard let resnetModel = resnetModel?.model else {
            fatalError("Unable to access model")
        }
        
        do {
            //Get model object
            let visionModel = try VNCoreMLModel(for: resnetModel)
            
            // Create vision request
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                
                if error != nil {
                    return
                }
                
                guard let observations = request.results else {
                    return
                }
                // Get the first prediction
                if let observation = observations.first as? VNClassificationObservation {
                    
                    // Print the confidence level * 100
                    print("Name \(observation.identifier) and confidence is \(observation.confidence * 100) %")
                    
                    DispatchQueue.main.async {
                        self.displayPredictions(text: observation.identifier, rayCastResult: rayCastResult)
                    }
                }
            }
            
            request.imageCropAndScaleOption = .centerCrop // if image is bigger, center crop it
            visionRequests = [request]
            
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored, options: [:])
            
            DispatchQueue.global().async {
                try? imageRequestHandler.perform(self.visionRequests)
            }
        }
        catch {
            print("Unable to create vision model")
        }
    }
}
