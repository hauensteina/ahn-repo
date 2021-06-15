//
//  AHXMain.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

/*
 This is the main entry point for the app. Add all functionality here.
 How it works:
 
 The SceneDelegate has a ContainerVC as root.
 ContainerVC has three VCs as children:
 (1) TopVC
 (2) MiddleVC
 (3) BottomVC
 - TopVC is meant for things like a Back button and a burger menu.
 - BottomVC is meant to be a tabbar where you click on Icons to go to
 different parts of the app.
 - MiddleVC is the Main window. It has the Navigation Controller.
 
 After startup, the SceneDelegate calls
 AHXMain.shared.main( navVC: rootVC.middleVC.nav) .
 This creates all view controllers for the main screen. They are immediately
 instantiated and layed out. They live in a dictionary AHXMain.shared.VCs .
 AHXMain.shared provides methods pushVC(), popVC(), topVC() to present
 ViewControllers from the VCs dictionary by name.
 
 The methods pushVC(), popVC(), topVC() are typically called when the user
 clicks something in BottomVC or TopVC. Depending on your app logic, actions
 in a presented VC (subVC of MiddleVC) might also call these methods.
 
 Implement your own VCs and add them to the VCs dictionary to build your app.
 Look at FirstVC and SecondVC to get a hang of how to write VCs.
 After that, it's all monkey see monkey do. Start by writing a ThirdVC and see
 if you can get it to show up and do something.
 
 */
 
import UIKit

//=====================
class AHXMain {
    static let shared = AHXMain()
    static var app = UIApplication.shared.delegate as! AppDelegate
    static var scene = SceneDelegate.shared!
    var navVC:UINavigationController!
    var VCs = [String:UIViewController]()
    var orderedVCs = [String]()
    
    // Entry Point to App, called from SceneDelegate
    //------------------------------------------------
    func main( navVC:UINavigationController) {
        self.navVC = navVC
        navVC.setNavigationBarHidden( true, animated:false)
        VCs = [
            "FirstVC": FirstVC(),
            "SecondVC": SecondVC(),
            "FontVC": FontVC()
        ]
        orderedVCs = [ "FirstVC", "SecondVC", "FontVC"]
        layoutVCs()
        pushVC( "FirstVC")
    } // main()
    
    // Make sure our VCs are instantiated, have correct size, and are layed out
    //----------------------------------------------------------------------------
    func layoutVCs() {
        let parentFrame = navVC.parent!.view.frame
        for vc in orderedVCs {
            let v = VCs[vc]!.view!
            AHL.left( v, parentFrame.minX)
            AHL.top( v, parentFrame.minY)
            AHL.width( v, parentFrame.width)
            AHL.height( v, parentFrame.height)
            VCs[vc]!.layout()
        }
    } // layoutVCs()
    
    // Push a VC by name
    //-------------------------------
    func pushVC( _ vcName:String) {
        let vc = VCs[vcName]!
        self.navVC.pushViewController( vc, animated: true)
        //ContainerVC.shared.bottomVC.selectButton( orderedVCs.firstIndex( of:vcName) ?? 0 )
    } // pushCV()

    // Pop top VC
    //----------------
    func popVC() {
        self.navVC.popViewController(animated: true)
    } // popCV()

    // Replace top VC by name
    //-------------------------------
    func topVC( _ vcName:String) {
        let vc = VCs[vcName]!
        self.navVC.popViewController( animated:true)
        self.navVC.pushViewController( vc, animated: true)
        //ContainerVC.shared.bottomVC.selectButton( orderedVCs.firstIndex( of:vcName) ?? 0 )
    } // topVC()
    
    // Get the currently active VC
    //----------------------------------------------
    class func currentVC() -> UIViewController {
        return self.shared.navVC.topViewController!
    } // currentVC()
    
} // class AHXMain
