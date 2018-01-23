//
//  LeftViewController.m
//  LGSideMenuControllerDemo
//

#import "Globals.h"
#import "CppInterface.h"
#import "LeftMenuController.h"
#import "LeftMenuCell.h"
#import "S3.h"
#import "TopViewController.h"
#import "UIViewController+LGSideMenuController.h"

enum {ITEM_NOT_SELECTED=0, ITEM_SELECTED=1};
enum {VIDEO_MODE=0, PHOTO_MODE=1, DEBUG_MODE=2};

@interface LeftMenuController ()
@property (strong, nonatomic) NSMutableArray *titlesArray;
@property UIFont *normalFont;
@property UIFont *selectedFont;
@property int mode; // debug, video, photo mode
@property NSMutableArray *s3_testcase_imgfiles;
@property NSMutableArray *s3_testcase_sgffiles;
@end

@implementation LeftMenuController

// Initialize left menu
//-----------------------
- (id)init
{
    self = [super initWithStyle:UITableViewStylePlain];
    if (self) {
        _selectedFont = [UIFont fontWithName:@"Verdana-Bold" size:16 ];
        _normalFont = [UIFont fontWithName:@"Verdana" size:16 ];
        NSArray *d = @[
                       @{ @"txt": @"Video Mode", @"state": @(ITEM_SELECTED) },
                       @{ @"txt": @"Photo Mode", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Debug Mode", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Edit Test Cases", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Add Test Case", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Run Test Cases", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Upload Test Cases", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Download Test Casess", @"state": @(ITEM_NOT_SELECTED) }
                       ];
        _titlesArray = [NSMutableArray new];
        for (NSDictionary *x in d) { [_titlesArray addObject:[x mutableCopy]]; }
        
        self.view.backgroundColor = [UIColor clearColor];

        [self.tableView registerClass:[LeftMenuCell class] forCellReuseIdentifier:@"cell"];
        self.tableView.separatorStyle =  UITableViewCellSeparatorStyleNone;
        self.tableView.contentInset = UIEdgeInsetsMake(44.0, 0.0, 44.0, 0.0);
        self.tableView.showsVerticalScrollIndicator = NO;
        self.tableView.backgroundColor = [UIColor clearColor];
    }
    return self;
} // init()

//-------------------------------
- (BOOL)prefersStatusBarHidden
{
    return YES;
}
//--------------------------------------------
- (UIStatusBarStyle)preferredStatusBarStyle
{
    return UIStatusBarStyleDefault;
}
//-----------------------------------------------------------
- (UIStatusBarAnimation)preferredStatusBarUpdateAnimation
{
    return UIStatusBarAnimationFade;
}

#pragma mark - UITableViewDataSource
//-----------------------------------------------------------------
- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView
{
    return 1;
}
//------------------------------------------------------------------------------------------
- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section
{
    return self.titlesArray.count;
}
//------------------------------------------------------------------------------------------------------
- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
    LeftMenuCell *cell = [tableView dequeueReusableCellWithIdentifier:@"cell"];
    cell.textLabel.text = self.titlesArray[indexPath.row][@"txt"];
    if ([_titlesArray[indexPath.row][@"state"] intValue] == ITEM_SELECTED) {
        [cell.textLabel setFont:_selectedFont];
    }
    else {
        [cell.textLabel setFont:_normalFont];
    }
    return cell;
} // cellForRowAtIndexPath()

#pragma mark - UITableViewDelegate
//-----------------------------------------------------------------------------------------------
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return 50;
}

// Unselect all menu Items
//---------------------------
- (void)unselectAll
{
    for (id x in _titlesArray) { x[@"state"] = @(ITEM_NOT_SELECTED); }
}

// Select or unselect one menu item by row
//-------------------------------------------------
- (void)setState:(int)state forRow:(NSInteger)row
{
    _titlesArray[row][@"state"] = @(state);
} // setState: forRow:

// Select or unselect one menu item by title
//-------------------------------------------------
- (void)setState:(int)state forMenuItem:(NSString *)txt
{
    int idx = -1;
    for (NSDictionary *d in _titlesArray) {
        idx++;
        if ([d[@"txt"] hasPrefix:txt]) {
            _titlesArray[idx][@"state"] = @(state);
            break;
        }
    }
} // setState: forMenuItem:

- (bool) videoMode { return _mode == VIDEO_MODE; }
- (bool) photoMode { return _mode == PHOTO_MODE; }
- (bool) debugMode { return _mode == DEBUG_MODE; }

// Handle left menu choice
//--------------------------------------------------------------------------------------------
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    TopViewController *topViewController = (TopViewController *)self.sideMenuController;
    NSString *menuItem = _titlesArray[indexPath.row][@"txt"];

    do {
        if ([menuItem hasPrefix:@"Edit Test Cases"]) {
            [self mnuEditTestCases];
        }
        else if ([menuItem hasPrefix:@"Add Test Case"]) {
            [self mnuAddTestCase];
        }
        else if ([menuItem hasPrefix:@"Run Test Cases"]) {
            [self mnuRunTestCases];
        }
        else if ([menuItem hasPrefix:@"Video Mode"]) {
            if (_mode == VIDEO_MODE) break;
            [self unselectAll];
            [self setState:ITEM_SELECTED forRow:indexPath.row];
            _mode = VIDEO_MODE;
            g_app.mainVC.btnCam.hidden = NO;
            [g_app.mainVC.frameExtractor resume];
            g_app.mainVC.lbBottom.text = @"Point the camera at a Go board";
            [g_app.mainVC doLayout];
        }
        else if ([menuItem hasPrefix:@"Photo Mode"]) {
            if (_mode == PHOTO_MODE) break;
            [self unselectAll];
            [self setState:ITEM_SELECTED forRow:indexPath.row];
            _mode = PHOTO_MODE;
            g_app.mainVC.btnCam.hidden = NO;
            [g_app.mainVC.frameExtractor resume];
            g_app.mainVC.lbBottom.text = @"Take a photo of a Go board";
            [g_app.mainVC doLayout];
        }
        else if ([menuItem hasPrefix:@"Debug Mode"]) {
            if (_mode == DEBUG_MODE) break;
            [self gotoDebugMode];
        }
        else if ([menuItem hasPrefix:@"Upload Test Cases"]) {
            [self mnuUploadTestCases];
        }
        else if ([menuItem hasPrefix:@"Download Test Cases"]) {
            [self mnuDownloadTestCases_0];
        }
        [self.tableView reloadData];
    } while(0);
    [topViewController hideLeftViewAnimated:YES completionHandler:nil];
} // didSelectRowAtIndexPath()

//---------------------
- (void)gotoDebugMode
{
    [self unselectAll];
    [self setState:ITEM_SELECTED forMenuItem:@"Debug Mode"];
    _mode = DEBUG_MODE;
    g_app.mainVC.btnCam.hidden = YES;
    [g_app.mainVC doLayout];
    [g_app.mainVC debugFlow:true];
    [self.tableView reloadData];
} // gotoDebugMode()

//============================
// Handlers for menu choices
//============================

// Save current image and board position as png and sgf.
// Filenames are testcase_nnnnn.png|sgf.
// The new nnnnn is one higher than the largest one found in the
// file systm.
//---------------------------
- (void)mnuAddTestCase
{
    NSArray *testfiles = globFiles(@"", @TESTCASE_PREFIX, @"*.png");
    NSString *last = changeExtension( [testfiles lastObject], @"");
    NSArray *parts = [last componentsSeparatedByString: @"_"];
    int fnum = [[parts lastObject] intValue] + 1;
    
    // Save image
    NSString *fname = nsprintf( @"%@%05d.png", @TESTCASE_PREFIX, fnum);
    fname = getFullPath( fname);
    [g_app.mainVC.cppInterface save_small_img:fname];
    
    // Save SGF
    fname = nsprintf( @"%@%05d.sgf", @TESTCASE_PREFIX, fnum);
    fname = getFullPath( fname);
    NSString *title = nsprintf( @"Testcase %d", fnum);
    [g_app.mainVC.cppInterface save_current_sgf:fname withTitle:title];
    
    [g_app.editTestCaseVC refresh];
    popup( nsprintf( @"Image added as Test Case %d", fnum), @"");
} // mnuAddTestCase()

// Show test cases from filesystem in a tableview, pick one.
//-------------------------------------------------------------
- (void) mnuEditTestCases
{
    [g_app.navVC pushViewController:g_app.editTestCaseVC animated:YES];
} // mnuSetCurrentTestCase()

// Run all test cases
//-------------------------
- (void)mnuRunTestCases
{
    NSArray *testfiles = globFiles(@"", @TESTCASE_PREFIX, @"*.png");
    NSMutableArray *errCounts = [NSMutableArray new];
    for (id fname in testfiles ) {
        NSString *fullfname = getFullPath( fname);
        // Load the image
        UIImage *img = [UIImage imageWithContentsOfFile:fullfname];
        // Load the sgf
        fullfname = changeExtension( fullfname, @".sgf");
        NSString *sgf = [NSString stringWithContentsOfFile:fullfname encoding:NSUTF8StringEncoding error:NULL];
        // Classify
        int nerrs = [g_app.mainVC.cppInterface runTestImg:img withSgf: sgf];
        [errCounts addObject:@(nerrs)];
    } // for
    // Show error counts in a separate View Controller
    UITextView *tv = g_app.testResultsVC.tv;
    NSMutableString *msg = [NSMutableString new];
    [msg appendString:@"Error Count by File\n"];
    [msg appendString:@"===================\n\n"];
    int i = -1;
    for (id fname in testfiles) {
        i++;
        long count = [errCounts[i] integerValue];
        NSString *line = nsprintf( @"%@:\t%5ld\n", fname, count);
        [msg appendString:line];
    }
    tv.text = msg;
    [g_app.navVC pushViewController:g_app.testResultsVC animated:YES];
} // mnuRunTestCases()

// Upload test cases to S3
//----------------------------
- (void)mnuUploadTestCases
{
    int idx;
    NSArray *testfiles;
    
    testfiles = globFiles(@TESTCASE_FOLDER, @TESTCASE_PREFIX, @"*.png");
    idx = -1;
    for (id fname in testfiles ) {
        idx++;
        S3_upload_file( nsprintf( @"%@/%@", @TESTCASE_FOLDER, fname) , ^(NSError *err) {});
    } // for
    testfiles = globFiles(@TESTCASE_FOLDER, @TESTCASE_PREFIX, @"*.sgf");
    NSInteger fcount = [testfiles count];
    idx = -1;
    for (id fname in testfiles ) {
        idx++;
        S3_upload_file( nsprintf( @"%@/%@", @TESTCASE_FOLDER, fname), 
                       ^(NSError *err) {
                           if (idx == fcount - 1) {
                               popup( @"Testcases uploaded", @"");
                           }
                       });
    } // for
} // mnuUploadTestCases()

// Download test cases from S3
//===============================

// Get list of png files
//--------------------------------
- (void)mnuDownloadTestCases_0
{
    _s3_testcase_imgfiles = [NSMutableArray new];
    S3_glob( nsprintf( @"%@/%@", @TESTCASE_FOLDER, @TESTCASE_PREFIX), @".png", _s3_testcase_imgfiles,
            ^(NSError *err) {
                if (err) {
                    popup( @"Failed to get S3 keys for img files", @"");
                }
                else {
                    [self mnuDownloadTestCases_1];
                }
            });
} // mnuDownloadTestCases_0()

// Get list of sgf files
//-------------------------------
- (void)mnuDownloadTestCases_1
{
    _s3_testcase_sgffiles = [NSMutableArray new];
    S3_glob( nsprintf( @"%@/%@", @TESTCASE_FOLDER, @TESTCASE_PREFIX), @".sgf", _s3_testcase_sgffiles,
            ^(NSError *err) {
                if (err) {
                    popup( @"Failed to get S3 keys for sgf files", @"");
                }
                else {
                    [self mnuDownloadTestCases_2];
                }
            });
} // mnuDownloadTestCases_1()

// Download image files
//-------------------------------
- (void)mnuDownloadTestCases_2
{
    int idx = -1;
    NSInteger fcount = [_s3_testcase_imgfiles count];
    if (!fcount) {
        popup( @"No testcases found.", @"");
        return;
    }
    for (id fname in _s3_testcase_imgfiles ) {
        idx++;
        S3_download_file( fname, fname,
                         ^(NSError *err) {
                             if (idx == fcount - 1) {
                                 [self mnuDownloadTestCases_3];
                             }
                         });
    } // for
} // mnuDownloadTestCases_2()

// Download sgf files
//-------------------------------
- (void)mnuDownloadTestCases_3
{
    int idx = -1;
    NSInteger fcount = [_s3_testcase_sgffiles count];
    if (!fcount) {
        popup( @"No sgf files found.", @"");
        return;
    }
    for (id fname in _s3_testcase_sgffiles ) {
        idx++;
        S3_download_file( fname, fname,
                         ^(NSError *err) {
                             if (idx == fcount - 1) {
                                 popup( @"Testcases downloaded", @"");
                             }
                         });
        
    } // for
} // mnuDownloadTestCases_3()

@end

































































