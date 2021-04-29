//
//  AHXConstants.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

class AHXConstants {
    // Dimensions of the whole display
    static let inset = AHXMain.scene.win.safeAreaInsets
    static let w = UIScreen.main.bounds.width
    static let h = UIScreen.main.bounds.height - inset.top - inset.bottom
    static let bottom = UIScreen.main.bounds.height - inset.bottom
    static let top = inset.top
    
    // Heights of top and bottom nav areas
    static let top_nav_height = 0.1 * h
    static let bottom_nav_height = 0.1 * h
    
    // Height of the middle (main) screen area
    static let main_height = h - top_nav_height - bottom_nav_height
    static let main_top = top + top_nav_height
    
    // Global layout constants
    static let btnHeight = 0.05 * h
    static let lmarg = 0.05 * w
    static let rmarg = 0.05 * w
    static let bmarg = 0.05 * h
    static let tmarg = 0.05 * h
    
    // Colors
    static let bgcol = AHU.RGB( 120,120,120)
    
} // class AHXConstants

typealias AHC = AHXConstants
