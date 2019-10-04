import React from "react";
import * as PropTypes from "prop-types";
import Button from "../../common/Button";
import CardContent from "@material-ui/core/es/CardContent/CardContent";
import CardHeader from "@material-ui/core/es/CardHeader/CardHeader";
import CardMenu from "../../common/CardMenu";
import DeleteIcon from "@material-ui/icons/DeleteRounded";
import Dialog from "@material-ui/core/es/Dialog/Dialog";
import DialogTitle from "@material-ui/core/es/DialogTitle/DialogTitle";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/es/DialogContentText/DialogContentText";
import DialogActions from "@material-ui/core/es/DialogActions/DialogActions";
import Divider from "@material-ui/core/es/Divider/Divider";
import Grid from "@material-ui/core/Grid";
import IconTypography from "../../common/IconTypography";
import Lorem from "react-lorem-component";
import RoundCornerAvatar from "../../common/RoundCornerAvatar";
import ShareIcon from "@material-ui/icons/ShareRounded";
import Typography from "@material-ui/core/es/Typography/Typography";
import withStyles from "@material-ui/core/styles/withStyles";
import Route from "react-router-dom/Route";

const styles = theme => ({
    root: {
        width: "85%",
        margin: "auto",
    },
    snapshot: {
        overflowX: "scroll",
        flexWrap: "nowrap",
    },
    snapshotContent: {
        flexShrink: 0,
    },
    description: {
        width: 342,
        height: "100%",
    },
    dialogPaper: {
        backgroundColor: theme.palette.grey[200],
        opacity: 0.98,
    }
});

class ModelDetails extends React.PureComponent {

    state = {
        shareDialog: false,
        deleteDialog: false,
    };

    id = "";

    menuItems = [
        {
            id: <IconTypography text={"share"} textProps={{variant: "subtitle1"}} icon={ShareIcon}/>,
            action: () => this.handleDialog("shareDialog", true),
        },
        {
            id: <IconTypography text={"delete"} textProps={{variant: "subtitle1", color: "error"}} icon={DeleteIcon}
                                iconProps={{color: "error"}}/>,
            action: () => this.handleDialog("deleteDialog", true),
        },
    ];

    handleDelete = history => {
        this.handleDialog("deleteDialog", false);
        this.props.deleteModel(this.id, () => history.push('./'));
    };

    componentWillMount() {
        const {location, fetchModel} = this.props;
        const params = new URLSearchParams(location.search);
        this.id = params.get("id");
        fetchModel(this.id);
    }

    handleDialog = (key, value) => {
        this.setState({[key]: value});
    };

    render() {
        const {classes, model} = this.props;
        const {shareDialog, deleteDialog} = this.state;

        return (
            <div className={classes.root}>
                {/* model header*/}
                <CardHeader action={<CardMenu menuItems={this.menuItems} fontSize={"large"}/>}
                            title={
                                <Grid spacing={40} container>
                                    <Grid item>
                                        <RoundCornerAvatar src={model.cover || "https://picsum.photos/224/?random"}
                                                           size={"large"}/>
                                    </Grid>
                                    <Grid item>
                                        <Grid className={classes.description} direction={'column'} spacing={16}
                                              container>
                                            <Grid item xs>
                                                <Typography variant={"h4"} gutterBottom>{model.name}</Typography>
                                                <Typography variant={"subtitle1"} color={"textSecondary"}
                                                            gutterBottom>{model.developer}</Typography>
                                            </Grid>
                                            <Grid>
                                                <Button variant={"outlined"} color={"primary"}
                                                        size={"large"} disableFocusRipple
                                                        disableRipple>Installed</Button>
                                            </Grid>
                                        </Grid>
                                    </Grid>
                                </Grid>
                            }/>

                <Divider/>
                {/* screen shot */}
                <CardContent>
                    <Typography variant={"h6"} gutterBottom>Screenshots</Typography>
                    <Grid className={classes.snapshot} spacing={8} container> {
                        [0, 1, 2, 3, 4].map(index => (
                            <Grid key={index} item>
                                <Typography className={classes.snapshotContent} component={'img'}
                                            src={`https://picsum.photos/200/300?image=${Math.round(Math.random() * 200)}`}/>
                            </Grid>
                        ))}
                    </Grid>
                </CardContent>
                {/* model description*/}
                <CardContent>
                    <Typography variant={"h6"} gutterBottom>Description</Typography>
                    <Typography variant={"body2"} component={props => <Lorem {...props}/>} paragraph/>
                </CardContent>
                {/* Information */}
                <CardContent>
                    <Typography variant={"h6"} gutterBottom>Information</Typography>
                    <Typography variant={"body2"} component={props => <Lorem {...props}/>}
                                sentenceLowerBound={1} sentenceUpperBound={2}
                                paragraphLowerBound={1} paragraphUpperBound={2} paragraph/>
                </CardContent>

                {/* share dialog */}
                <Dialog classes={{paper: classes.dialogPaper}} open={shareDialog}
                        onClose={() => this.handleDialog("shareDialog", false)}>
                    <DialogTitle>{`Share Model ${model.name}`}</DialogTitle>
                    <DialogContent>
                        <DialogContentText>
                            Cras mattis consectetur purus sit amet fermentum. Cras justo odio, dapibus ac
                            facilisis in, egestas eget quam.
                        </DialogContentText>
                        <DialogActions>
                            <Button color={"primary"} onClick={() => this.handleDialog("shareDialog", false)}>
                                Ok
                            </Button>
                        </DialogActions>
                    </DialogContent>
                </Dialog>

                {/* delete dialog */}
                <Dialog classes={{paper: classes.dialogPaper}} open={deleteDialog}
                        onClose={() => this.handleDialog("deleteDialog", false)}>
                    <DialogTitle>{`Delete Model "${model.name}"?`}</DialogTitle>
                    <DialogContent>
                        <DialogContentText>
                            Deleting model {model.name} will also delete all its data.
                            This action cannot be undo.
                        </DialogContentText>
                        <DialogActions>
                            <Button color={"primary"} onClick={() => this.handleDialog("deleteDialog", false)}>
                                Cancel
                            </Button>

                            {/* Delete button */}
                            <Route render={({history}) =>
                                <Button onClick={() => this.handleDelete(history)}>
                                    <Typography color={"error"}>Delete Anyway</Typography>
                                </Button>
                            }/>

                        </DialogActions>
                    </DialogContent>
                </Dialog>
            </div>
        );
    }
}

ModelDetails
    .propTypes = {
    classes: PropTypes.object.isRequired,
    location: PropTypes.shape({
        search: PropTypes.string.isRequired,
    }).isRequired,
    model: PropTypes.object,
    fetchModel: PropTypes.func.isRequired,
    deleteModel: PropTypes.func.isRequired,
};

export default withStyles(styles)(ModelDetails);
