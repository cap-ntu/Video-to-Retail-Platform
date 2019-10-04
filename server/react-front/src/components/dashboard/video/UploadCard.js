import React from "react";
import * as PropTypes from "prop-types";
import Card from "@material-ui/core/es/Card/Card";
import CardHeader from "@material-ui/core/CardHeader/CardHeader";
import IconButton from "@material-ui/core/IconButton/IconButton";
import CloseIcon from "@material-ui/icons/Close";
import CardContent from "@material-ui/core/es/CardContent/CardContent";
import Button from "../../common/Button";
import CheckBox from "@material-ui/core/Checkbox";
import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import FormControlLabel from "@material-ui/core/es/FormControlLabel/FormControlLabel";
import FormControl from "@material-ui/core/FormControl";
import Typography from "@material-ui/core/Typography";
import Collapse from "@material-ui/core/Collapse";
// File pond
import {FilePond} from "react-filepond";
import Switch from "@material-ui/core/es/Switch/Switch";
import Link from "../../common/Link";

const styles = theme => ({
    root: {
        maxWidth: 720,
        margin: 'auto',
    },
    roundCardAction: {
        borderRadius: theme.shape.borderRadius,
    },
    upload: {
        width: 224,
        height: 224,
        opacity: 0.75,
        transition: 'color 1.1s cubic-bezier(.17,.63,.61,1.06)',
    },
    uploadDragover: {
        backgroundColor: theme.palette.primary,
    },
    uploadInput: {
        position: 'absolute',
        width: '100%',
        height: '100%',
        opacity: 0,
        cursor: 'pointer',
    },
    formRoot: {
        display: 'flex',
    },
    modelSelectionGrid: {
        paddingBottom: 2 * theme.spacing.unit,
        '&:last-child': {
            paddingBottom: 0,
        }
    }
});

const types = ["OBJECT", "FACE", "TEXT", "SCENE"];

class UploadCard extends React.Component {

    state = {
        files: [],
        process: true,
        advanced: false,
        formLock: 0, // modifiable of model form
        modelMutex: false,
        defaultModelMutex: false,
        modelTypes: {},
    };

    componentWillMount() {
        const {fetchModelList, fetchDefaultModelList} = this.props;
        fetchDefaultModelList(() => this.setState({modelMutex: true}));
        fetchModelList(() => this.setState({defaultModelMutex: true}));
    }

    componentDidUpdate(prevProps, prevState, snapshot) {

        const {models, defaultModels} = this.props;
        const {modelTypes, modelMutex, defaultModelMutex} = this.state;

        if (modelMutex && defaultModelMutex && Object.keys(models).length && Object.keys(defaultModels).length) {
            types.forEach(key => {
                modelTypes[key] = {};
                models[key].forEach(model => {
                    modelTypes[key][model.name] = Object.keys(defaultModels).length > 0 && defaultModels[key].some(_model => _model.name === model.name)
                })
            });

            this.setState({modelTypes: modelTypes, modelMutex: false, defaultModelMutex: false});
        }
    }

    handleChange = name => event => {
        this.setState({[name]: event.target.checked});
    };

    handleChangeModel = (key, name) => {
        return event => {
            const checked = event.target.checked;

            this.setState(state => ({
                ...state,
                modelTypes: {...state.modelTypes, [key]: {...state.modelTypes[key], [name]: checked}}
            }))
        };
    };

    handleClickUploadBtn = () => {
        this.pond.processFiles();
    };

    handleUpload = (_1, file, metadata, loadListener, errorListener, progressListener, abortListener) => {

        const {postVideo} = this.props;
        const {process, modelTypes} = this.state;

        postVideo({file, process, modelTypes, loadListener, errorListener, progressListener, abortListener});
    };

    render() {
        const {classes, models} = this.props;
        const {process, advanced, formLock, modelTypes} = this.state;

        return (
            <Card className={classes.root}>
                <CardHeader action={
                    <IconButton component={props => <Link {...props}/>} to={'./'}>
                        <CloseIcon/>
                    </IconButton>
                }/>
                <CardContent>

                    {/* File uploader */}
                    <FilePond ref={ref => this.pond = ref}
                              allowMultiple
                              instantUpload={false}
                              server={{process: this.handleUpload}}
                              maxFiles={2}
                              fileRenameFunction={file => new Promise(resolve => {
                                  resolve(window.prompt('Enter new filename', file.name))
                              })}
                              onupdatefiles={(fileItems) => {
                                  this.setState({
                                      files: fileItems.map(fileItem => fileItem.file)
                                  });
                              }}
                              acceptedFileTypes={['video/mp4']}
                              onprocessfilestart={() => this.setState({formLock: formLock - 1})}
                              onprocessfile={() => this.setState({formLock: formLock + 1})}>
                    </FilePond>

                    {/* Process selection control */}
                    <Typography component='form' align='center'>
                        <FormControl component="fieldset" disabled={formLock < 0}>
                            <FormControlLabel
                                control={<CheckBox checked={process}
                                                   onChange={this.handleChange('process', true)}
                                                   value="process"/>}
                                label={"process"}/>
                        </FormControl>
                    </Typography>

                    {/* Model selection panel */}
                    <Collapse in={advanced && process} timeout="auto" unmountOnExit>
                        <CardContent> {
                            Object.keys(modelTypes).length ?
                                <FormControl component="fieldset"> {
                                    types.map(key => (
                                        <React.Fragment key={key}>
                                            <Typography gutterBottom>
                                                {key}
                                            </Typography>
                                            <Grid className={classes.modelSelectionGrid} container spacing={8}>{
                                                models[key].map(model => (
                                                    <Grid key={model.name} item>
                                                        <FormControlLabel
                                                            control={
                                                                <Switch checked={modelTypes[key][model.name]}
                                                                        onChange={this.handleChangeModel(key, model.name)}/>}
                                                            label={model.name}
                                                            disabled={formLock < 0}/>
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </React.Fragment>
                                    ))}
                                </FormControl> :
                                <Typography align={"center"} color={'default'}>
                                    No model available now
                                </Typography>
                        }
                        </CardContent>
                    </Collapse>

                    {/* Advance-model-selection and Upload buttons */}
                    <Grid container direction="row">
                        <Button onClick={() => this.setState({advanced: !advanced})}
                                variant={advanced ? "outlined" : null}
                                color={advanced ? "primary" : null}
                                disabled={!process}>
                            Advanced
                        </Button>
                        <div style={{flexGrow: 1}}/>
                        <Button
                            variant="text"
                            color="primary"
                            onClick={this.handleClickUploadBtn}
                            disabled={formLock < 0}>
                            Upload
                        </Button>
                    </Grid>

                </CardContent>
            </Card>
        )
    }
}

UploadCard.propTypes = {
    classes: PropTypes.object.isRequired,
    models: PropTypes.shape({
        [PropTypes.oneOf(["OBJECT", "TEXT", "FACE", "SCENE"])]: PropTypes.arrayOf(
            PropTypes.shape({
                name: PropTypes.string.isRequired,
                id: PropTypes.string.isRequired,
            }).isRequired,
        ),
    }).isRequired,
    defaultModels: PropTypes.shape({
        [PropTypes.oneOf(["OBJECT", "TEXT", "FACE", "SCENE"])]: PropTypes.arrayOf(
            PropTypes.shape({
                name: PropTypes.string.isRequired,
                id: PropTypes.string.isRequired,
            }).isRequired,
        ),
    }).isRequired,
    fetchModelList: PropTypes.func.isRequired,
    fetchDefaultModelList: PropTypes.func.isRequired,
    postVideo: PropTypes.func.isRequired,
};

export default withStyles(styles)(UploadCard);
